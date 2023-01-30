#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import torch


class NaiveRepeatLast(torch.nn.Module):
    def __init__(self):
        """Returns prediction consisting of repeating last frame."""
        super(NaiveRepeatLast, self).__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = x[:, 11:12, :, :]
        x = torch.repeat_interleave(x, repeats=6, dim=1)
        return x
