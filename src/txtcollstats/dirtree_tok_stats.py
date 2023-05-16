import os
import random
import sys
import statistics
from dataclasses import dataclass
from typing import List
import fnmatch

import tiktoken
import numpy as np

ADA_MAX_TOKENS = 8191

EXCLUDE_PATTERNS = [ "*.bz2" ]


def list_directory_tree(dir_, exclude_patterns=EXCLUDE_PATTERNS):
    if exclude_patterns is None:
        exclude_patterns = []

    for dirpath, _, filenames in os.walk(dir_):
        for filename in filenames:
            if not any(fnmatch.fnmatch(filename, pattern) 
                       for pattern in exclude_patterns):
                yield os.path.join(dirpath, filename)


def sample_filenames(dir_, n):
    return random.sample(list(list_directory_tree(dir_)), n)


def load_files(filenames, char_encoding='utf-8'):
    for filename in filenames:
        with open(filename, encoding=char_encoding) as f:
            yield f.read()


def count_tokens(text, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def sample_token_lengths(dir_, n, char_encoding='utf-8'):
    filenames = sample_filenames(dir_, n)
    texts = load_files(filenames, char_encoding=char_encoding)
    return [count_tokens(text) for text in texts]


@dataclass
class SummaryStatistics:
    collname: str
    nsmpl: int
    mean: float
    median: float
    stdev: float
    min: int
    max: int
    percentile_05:  float
    percentile_95:  float
    above_threshold : int 
 
    threshold : int = ADA_MAX_TOKENS

    @classmethod
    def from_int_seq(cls, collname, seq, threshold=ADA_MAX_TOKENS):
        return cls(
            collname=collname,
            nsmpl=len(seq),
            mean=statistics.mean(seq),
            median=statistics.median(seq),
            stdev=statistics.stdev(seq),
            min=min(seq),
            max=max(seq),
            percentile_05=np.percentile(seq, 5),
            percentile_95=np.percentile(seq, 95),
            above_threshold=sum(1 for x in seq if x > threshold),
            threshold=threshold
        )

    def formatted(self):
        return f"""\
nsmpl:  {self.nsmpl:5d}
mean:   {self.mean:5.0f}
stdev:  {self.stdev:5.0f}
min:    {self.min:5d}
 5%:    {self.percentile_05:5.0f}
median: {self.median:5.0f}
95%:    {self.percentile_95:5.0f}
max:    {self.max:5d}
>{self.threshold:4d}:  {self.above_threshold:5d}%
"""


def summary_statistics(collname, token_lengths):
    return SummaryStatistics.from_int_seq(collname, token_lengths)


def smy_stats_list_markdown_table(stats_list : List[SummaryStatistics]):
    return f"""\
| Collection | nsmpl | mean | stdev | min| 5%  | median | 95% | max | >8191 |
|------------|-------|------|-------|----|-----|--------|-----|-----|-------|
{chr(10).join([f"| {s.collname} | {s.nsmpl} | {s.mean:.0f} | {s.stdev:.0f} | {s.min} | {s.percentile_05:.0f} | {s.median:.0f} | {s.percentile_95:.0f} | {s.max} | {s.above_threshold} |" for s in stats_list])}
"""


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('coll_list_file')
    parser.add_argument('--n', type=int, default=10000)

    args = parser.parse_args()
    stats_list = []
    for coll_line in open(args.coll_list_file):
        info = coll_line.strip().split(':')
        collname = info[0]
        dir_ = info[1]
        if len(info) > 2:
            char_encoding = info[2]
        else:
            char_encoding = 'utf-8'
        token_lengths = sample_token_lengths(dir_, args.n, char_encoding=char_encoding)
        stats_list.append(summary_statistics(collname, token_lengths))
    print(smy_stats_list_markdown_table(stats_list))
