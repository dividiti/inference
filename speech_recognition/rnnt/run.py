# Copyright 2020 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import mlperf_loadgen as lg
import subprocess

import os
from pathlib import Path

RNNT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
MLPERF_CONF = RNNT_DIR / "../../mlperf.conf"
MLPERF_CONF = MLPERF_CONF.resolve()
USER_CONF = RNNT_DIR / "user.conf"
USER_CONF = USER_CONF.resolve()
PYTHON = os.getenv('CK_ENV_COMPILER_PYTHON_FILE', 'python3')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--mlperf_conf", default=str(MLPERF_CONF), help="mlperf rules config")
    parser.add_argument("--user_conf", default=str(USER_CONF), help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--pytorch_config_toml", default="pytorch/configs/rnnt.toml")
    parser.add_argument("--pytorch_checkpoint", default="pytorch/work_dir/rnnt.pt")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--perf_count", type=int, default=None)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--instr", action="store_true", help="enable instrumentation", default=False)
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    args = get_args()

    if args.backend == "pytorch":
        from pytorch_SUT import PytorchSUT
        sut = PytorchSUT(args.pytorch_config_toml, args.pytorch_checkpoint,
                         args.dataset_dir, args.manifest, args.perf_count, args.instr)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "rnnt", args.scenario)
    settings.FromConfig(args.user_conf, "rnnt", args.scenario)

    if args.accuracy or args.instr:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_path = args.log_dir
    os.makedirs(log_path, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    print("Running Loadgen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)

    setinstr = '--instr' if args.instr else ''
    if args.accuracy or args.instr:
        cmd = f"{PYTHON} {RNNT_DIR}/accuracy_eval.py --log_dir {log_path} --dataset_dir {args.dataset_dir} --manifest {args.manifest} " + setinstr
        print(f"Running accuracy script: {cmd}")
        subprocess.check_call(cmd, shell=True)

    sut.dump_instr()

    print("Done!")


if __name__ == "__main__":
    main()
