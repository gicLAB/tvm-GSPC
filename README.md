<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM-GSPC: Optimizing Grouped Convolutions on Edge Devices

Modified TVM codebase to add support for Grouped Spatial Pack Convolution (GSPC).

Testing in a Debian 10 GNU/Linux Bash environment.

To run a TVM workload using GSPC on x86 and ARM platforms, set the environment variable `TVM_USE_GSPC` to true.  For remote devices, this codebase, or `master` TVM commit `95e06b3ec9` are known to work.

For installation details [see the TVM documentation](https://docs.tvm.ai/install/from_source.html#install-from-source).



License
-------
© Contributors Licensed under an [Apache-2.0](LICENSE) license.

