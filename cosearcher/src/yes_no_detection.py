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

from abc import ABC, abstractmethod

import utils


class YesNoDetector(ABC):
    @abstractmethod
    def stance(self, answer: str) -> str:
        pass


class DummyYesNoDetector(YesNoDetector):
    def stance(self, answer: str) -> str:
        answer = utils.strip_punctuation(answer)
        sent = answer.lower().split()
        if "yes" in sent[:3]:
            return "yes"
        elif "no" in sent[:3]:
            return "no"
        return "unknown"


class AlwaysYesYesNoDetector(YesNoDetector):
    def stance(self, answer: str) -> str:
        return "yes"
        