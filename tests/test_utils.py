#  Copyright (c) 2023, The SmbRec Authors.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import unittest
import smbrec.utils as utils


class TestFileUtils(unittest.TestCase):
    """ utils.file_utils.py的测试用例 """

    def test_compress(self):
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf")
        dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        path = utils.file_utils.compress(src_path, dest_dir)
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
