from pypinyin import lazy_pinyin


def compare_pinyin(s1, s2):
    s1_pinyin = ''.join(lazy_pinyin(s1))
    s2_pinyin = ''.join(lazy_pinyin(s2))
    return s1_pinyin == s2_pinyin

