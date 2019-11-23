# Python version of the evaluation script from CoNLL'00-
# Originates from: https://github.com/spyysalo/conlleval.py


# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import sys
import re
import codecs
from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # 标注命名实体且被预测出的数量（词数）
        self.found_correct = 0    # 标注命名实体数量(词数)
        self.found_guessed = 0    # 预测命名实体数量（词数）

        self.correct_tags = 0     # 预测正确字符数（字数）
        self.token_counter = 0    # 总字符数_不含空格（字数）

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)     # 真实正例中的类别分类  
        self.t_found_guessed = defaultdict(int)     # 预测正例中的类别分类


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def evaluate(iterable, options=None):   # iterable是全体样本
                                        # 根据文件中结果填完EvalCounts()类
    if options is None:
        options = parse_args([])    # use defaults

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for line in iterable:                       # 每次取一个字及标签
        line = line.rstrip('\r\n')              # line: "字符 标注Tag 预测Tag

        if options.delimiter == ANY_SPACE:
            features = line.split()             # features: [字符, 标注Tag, 预测Tag]
        else:
            features = line.split(options.delimiter)

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())   # 预测tag guessed:O/B/I; guessed_type: LOC/PER/ORG
        correct, correct_type = parse_tag(features.pop())   # 正确tag correct:O/B/I; correct_type: LOC/PER/ORG
        first_item = features.pop(0)                        # first_item: 字符

        if first_item == options.boundary:                  # first_item是两句间的空白，不是一个字符
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)         # 通过标注tag判断上一词与该词之间是否是一个词结束
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)         # 通过预测tag判断上一词与该词之间是否是一个词结束
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)     # 通过标注tag判断上一词与该词之间是否是一个词开始
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)     # 通过预测tag判断上一词与该词之间是否是一个词开始

        if in_correct:                                                      # 标注tag和预测tag均表示该字为词头,且命名实体类型识别正确
            if (end_correct and end_guessed and last_guessed_type == last_correct_type):  # 若上一个字符是真实词尾且模型正确预测（类型也正确）
                in_correct = False
                counts.correct_chunk += 1                                   # 真正立数+1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):  # 若该字符预测错误
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:    # 本字符是真实词头且预测正确（类型也正确）
            in_correct = True

        if start_correct:                                       # 根据标注tag判断为词头
            counts.found_correct += 1                           # 真实命名实体数+1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:                                       # 根据预测tag判断为词头
            counts.found_guessed += 1                           # 预测命名实体数+1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:                      # first_item不是两句间的空白，是一个字符
            if correct == guessed and guessed_type == correct_type:    # 标注tag与预测tag相同
                counts.correct_tags += 1                        # 预测正确字数+1
            counts.token_counter += 1                           # 总字数（不含空格）+1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:                                      # 若最后一个字符在真正例词且是预测正例词中
        counts.correct_chunk += 1                       # 真实正例且预测正确数+1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts


def uniq(iterable):     # 去掉重复元素，返回列表中每个元素都是唯一
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct, guessed, total):     # correct:预测为正例且为真正例数; guessed:预测正例数; total:真正例数
    tp, fp, fn = correct, guessed-correct, total-correct    # 真正例、假正例、假反例
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)    # precise 精确率
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)    # recall 召回率
    f = 0 if p + r == 0 else 2 * p * r / (p + r)    # F1
    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts):
    c = counts
    overall = calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)  # 计算总的p r f
    by_type = {}
    for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):              
        by_type[t] = calculate_metrics(c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]) # 分别计算三类实体的p r f
    return overall, by_type  # overall: Metrics; by_type: {PER: Metrics, LOC: Metrics, ORG: Metrics}


def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' % (c.token_counter, c.found_correct)) # 总字数、预测正例词数
    out.write('found: %d phrases; correct: %d.\n' % (c.found_guessed, c.correct_chunk))     # 预测正例词数、真正例词数

    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' % (100.*c.correct_tags/c.token_counter))    # 计算准确率
        out.write('precision: %6.2f%%; ' % (100.*overall.prec))                     # 精确率
        out.write('recall: %6.2f%%; ' % (100.*overall.rec))                         # 召回率
        out.write('FB1: %6.2f\n' % (100.*overall.fscore))                           # F指数

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100.*m.prec))
        out.write('recall: %6.2f%%; ' % (100.*m.rec))
        out.write('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))


def report_notprint(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)  # 获得总的P、R、F1结果和三类实体分别的P、R、F1结果

    c = counts
    final_report = []
    line = []
    line.append('processed %d tokens with %d phrases; ' % (c.token_counter, c.found_correct))
    line.append('found: %d phrases; correct: %d.\n' % (c.found_guessed, c.correct_chunk))
    final_report.append("".join(line))

    if c.token_counter > 0:
        line = []
        line.append('accuracy: %6.2f%%; ' % (100.*c.correct_tags/c.token_counter))  # 准确率
        line.append('precision: %6.2f%%; ' % (100.*overall.prec))           # 精确率
        line.append('recall: %6.2f%%; ' % (100.*overall.rec))               # 召回率
        line.append('FB1: %6.2f\n' % (100.*overall.fscore))                 # F1指数
        final_report.append("".join(line))

    for i, m in sorted(by_type.items()):    # 按照LOC ORG PER的顺序遍历
        line = []
        line.append('%17s: ' % i)                                                   # LOC/ORG/PER
        line.append('precision: %6.2f%%; ' % (100.*m.prec))                         # 精确率
        line.append('recall: %6.2f%%; ' % (100.*m.rec))                             # 召回率
        line.append('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))     # F1指数 预测正例数
        final_report.append("".join(line))
    return final_report


def end_of_chunk(prev_tag, tag, prev_type, type_):          # 判断上一个字是否是单词的结尾
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):        # 判断上一个词是否是单词的头
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start


def return_report(input_file):
    with codecs.open(input_file, "r", "utf-8") as f:
        counts = evaluate(f)
    return report_notprint(counts)


def main(argv):
    args = parse_args(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, args)
    report(counts)

if __name__ == '__main__':
    sys.exit(main(sys.argv))