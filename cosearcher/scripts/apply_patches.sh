#!/bin/bash

#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.


patch src/thirdparty/lexvec.py < patches/lexvec.patch
patch src/thirdparty/ql/QL.py < patches/ql_QL.patch
patch src/thirdparty/ql/utils/utils.py < patches/ql_utils.patch
patch src/thirdparty/transformer/run_glue.py < patches/transformer_run_glue.patch
patch src/thirdparty/transformer/utils_glue.py < patches/transformer_utils_glue.patch