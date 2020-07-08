# Copyright (c) 2020, Cerebras Systems, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

import time
import json
import array
import torch
import numpy as np
import toml
import mlperf_loadgen as lg
from tqdm import tqdm

from QSL import AudioQSL, AudioQSLInMemory
from decoders import ScriptGreedyDecoder
from helpers import add_blank_label
from preprocessing import AudioPreprocessing
from model_separable_rnnt import RNNT


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


class PytorchSUT:
    def __init__(self, config_toml, checkpoint_path, dataset_dir,
                 manifest_filepath, perf_count, enable_instr):
        config = toml.load(config_toml)

        dataset_vocab = config['labels']['labels']
        rnnt_vocab = add_blank_label(dataset_vocab)
        featurizer_config = config['input_eval']

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries,
                                   self.process_latencies)
        self.qsl = AudioQSLInMemory(dataset_dir,
                                    manifest_filepath,
                                    dataset_vocab,
                                    featurizer_config["sample_rate"],
                                    perf_count)
        self.audio_preprocessor = AudioPreprocessing(**featurizer_config)
        self.audio_preprocessor.eval()
        self.audio_preprocessor = torch.jit.script(self.audio_preprocessor)
        self.audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
            torch._C._freeze_module(self.audio_preprocessor._c))

        model = RNNT(
            feature_config=featurizer_config,
            rnnt=config['rnnt'],
            num_classes=len(rnnt_vocab),
            instr=enable_instr
        )
        model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path),
                              strict=False)
        model.eval()
        model.hotswap_init()
        ''' #remove jitting of script as interferes with plugins
        model.encoder = torch.jit.script(model.encoder)
        model.encoder = torch.jit._recursive.wrap_cpp_module(
            torch._C._freeze_module(model.encoder._c))
        model.prediction = torch.jit.script(model.prediction)
        model.prediction = torch.jit._recursive.wrap_cpp_module(
            torch._C._freeze_module(model.prediction._c))
        model.joint = torch.jit.script(model.joint)
        model.joint = torch.jit._recursive.wrap_cpp_module(
            torch._C._freeze_module(model.joint._c))
        model = torch.jit.script(model)
        '''

        self.greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)

        self.instr = enable_instr
        self.samples = []

    def issue_queries(self, query_samples):
        query_count = 0
        for query_sample in query_samples:
            waveform = self.qsl[query_sample.index]
            assert waveform.ndim == 1
            waveform_length = np.array(waveform.shape[0], dtype=np.int64)
            waveform = np.expand_dims(waveform, 0)
            waveform_length = np.expand_dims(waveform_length, 0)
            with torch.no_grad():
                waveform = torch.from_numpy(waveform)
                waveform_length = torch.from_numpy(waveform_length)
                feature, feature_length = self.audio_preprocessor.forward((waveform, waveform_length))
                assert feature.ndim == 3
                assert feature_length.ndim == 1
                feature = feature.permute(2, 0, 1)

                batch_start = time.time()
                _, _, transcript = self.greedy_decoder.forward(feature, feature_length)
                batch_end = time.time()

                if self.instr:
                    pre, post, dec = self.greedy_decoder._model.instr.get_timings()
                    sample = {}
                    sample['exe_time'] = batch_end - batch_start
                    sample['qsl_idx'] = query_sample.index
                    sample['query_id'] = query_sample.id
                    sample['pre_time'] = pre
                    sample['post_time'] = post
                    sample['dec_time'] = dec

                    self.samples.append(sample)

            assert len(transcript) == 1
            response_array = array.array('q', transcript[0])
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_sample.id, bi[0],
                                              bi[1] * response_array.itemsize)
            lg.QuerySamplesComplete([response])

            query_count += 1

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        print("Average latency (ms) per query:")
        print(np.mean(latencies_ns)/1000000.0)
        print("Median latency (ms): ")
        print(np.percentile(latencies_ns, 50)/1000000.0)
        print("90 percentile latency (ms): ")
        print(np.percentile(latencies_ns, 90)/1000000.0)

    def dump_instr(self):
        if self.instr:
          with open("timing_instr.json", 'w') as instrfp:
              json.dump(self.samples, instrfp, indent=2)

    def __del__(self):
        lg.DestroySUT(self.sut)
        print("Finished destroying SUT.")
