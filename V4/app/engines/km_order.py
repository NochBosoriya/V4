"""Khmer word order for nouns: the noun comes first and what describes it
follows — ប្រធានថ្នាក់, "head (of the) class" — the opposite of Chinese and
English, which put the describing word first (班长 = class + head, "class
monitor"). A dub once said ថ្នាក់ប្រធាន for 班长: the AI kept the Chinese
order on a short, list-like line ("班长681，沈知夏").

    rule(src_lang)        the formula, for the translation prompts
    COMPOUNDS             common drama nouns: Chinese, Khmer noun, its describer
    hints()               those nouns as a prompt hint (glossary + translation)
    fix(source, text)     a known compound in the source's order, put right

No Qt imports.
"""
from __future__ import annotations

import re

# (Chinese, the Khmer noun, what describes it): right = noun + describer
COMPOUNDS: list[tuple[str, str, str]] = [
    ("副班长", "អនុប្រធាន", "ថ្នាក់"),
    ("班长", "ប្រធាន", "ថ្នាក់"),
    ("副校长", "នាយករង", "សាលា"),
    ("校长", "នាយក", "សាលា"),
    ("班主任", "គ្រូ", "បន្ទុកថ្នាក់"),
    ("学生会主席", "ប្រធាន", "សមាគមសិស្ស"),
    ("保安队长", "ប្រធាន", "សន្តិសុខ"),
    ("队长", "ប្រធាន", "ក្រុម"),
    ("组长", "ប្រធាន", "ក្រុម"),
    ("村长", "មេ", "ភូមិ"),
    ("市长", "អភិបាល", "ក្រុង"),
    ("行长", "នាយក", "ធនាគារ"),
    ("店长", "អ្នកគ្រប់គ្រង", "ហាង"),
    ("酒店经理", "អ្នកគ្រប់គ្រង", "សណ្ឋាគារ"),
    ("公司老板", "ថៅកែ", "ក្រុមហ៊ុន"),
    ("董事长", "ប្រធាន", "ក្រុមហ៊ុន"),
    ("总分", "ពិន្ទុ", "សរុប"),
    ("满分", "ពិន្ទុ", "ពេញ"),
    ("同学", "មិត្ត", "រួមថ្នាក់"),
    ("同桌", "មិត្ត", "រួមតុ"),
    ("学霸", "សិស្ស", "ពូកែ"),
    ("学渣", "សិស្ស", "ខ្សោយ"),
    ("新生", "សិស្ស", "ថ្មី"),
    ("校花", "ផ្កា", "សាលា"),
    ("英语老师", "គ្រូ", "ភាសាអង់គ្លេស"),
    ("数学老师", "គ្រូ", "គណិតវិទ្យា"),
    ("手机", "ទូរស័ព្ទ", "ដៃ"),
]
# a Khmer word that starts with a compound's last part: not that compound
# (សាលានាយកដ្ឋាន is a "department hall", ប្រធានាធិបតី a president)
_LONGER = ("ដ្ឋាន", "ធិបតី", "ការ")
_ZWSP = "​"


def rule(src_lang: str = "") -> str:
    """The formula for Khmer noun phrases, with examples from the source
    language (Chinese or English)."""
    base = (src_lang or "").split("-")[0].lower()
    if base in ("zh", "ja", "yue"):
        ex = ("班长 → ប្រធានថ្នាក់ (never ថ្នាក់ប្រធាន); 校长 → នាយកសាលា; 总分 → "
              "ពិន្ទុសរុប; 英语老师 → គ្រូភាសាអង់គ្លេស; 红衣服 → អាវក្រហម; 我妈 → "
              "ម៉ាក់ខ្ញុំ; 他的手机 → ទូរស័ព្ទរបស់គាត់; 三个人 → មនុស្សបីនាក់; 这本书 → "
              "សៀវភៅនេះ; 我买的车 → ឡានដែលខ្ញុំទិញ; 第一名 → លេខមួយ; 王老师 → "
              "លោកគ្រូវ៉ាង; 张总 → លោកប្រធានចាង; 川哥 → បងឈួន")
    else:
        ex = ("class president → ប្រធានថ្នាក់ (never ថ្នាក់ប្រធាន); school principal → "
              "នាយកសាលា; total score → ពិន្ទុសរុប; English teacher → គ្រូភាសាអង់គ្លេស; "
              "the red car → ឡានពណ៌ក្រហម; my mom → ម៉ាក់ខ្ញុំ; his phone → "
              "ទូរស័ព្ទរបស់គាត់; three people → មនុស្សបីនាក់; this book → សៀវភៅនេះ; "
              "the car I bought → ឡានដែលខ្ញុំទិញ; first place → លេខមួយ; Mr. Wang → "
              "លោកវ៉ាង; Teacher Wang → លោកគ្រូវ៉ាង")
    return ("Khmer word order for nouns — the noun FIRST, then what describes it: "
            "noun + its kind / material / quality + whose or where + number and "
            "classifier + នេះ / នោះ + ដែល … (\"the head of the class\", not "
            "\"class head\"). The original puts the describing words first: turn "
            "them around: " + ex + ". An ordinal is noun + ទី + number (ថ្ងៃទីមួយ). "
            "A title or a kinship word comes before the name. Only Khmer's own "
            "prefixes stay in front: អនុ (vice-), អគ្គ (general), អតីត (former), "
            "មហា (great) — អនុប្រធាន, អគ្គនាយក, អតីតសង្សារ, មហាសេដ្ឋី; រង "
            "(deputy) follows: នាយករង.")


def hints() -> str:
    """The common drama nouns, for the glossary and the translation."""
    return ("Chinese school and work nouns in Khmer (the noun first): "
            + "; ".join(f"{zh} → {head}{desc}" for zh, head, desc in COMPOUNDS) + ".")


def fix(source: str, text: str) -> tuple[str, int]:
    """`text` with every known compound of `source` that is written in the
    source's order (ថ្នាក់ប្រធាន for 班长) put in Khmer order (ប្រធានថ្នាក់);
    (text, how many were put right). Only compounds the original says, and
    only whole words (not the start of a longer Khmer word)."""
    if not text or not source:
        return text, 0
    n = 0
    taken: set[str] = set()
    for zh, head, desc in COMPOUNDS:
        if zh not in source or any(zh in t for t in taken):
            continue
        taken.add(zh)
        wrong = re.compile(re.escape(desc) + f"[ {_ZWSP}]?" + re.escape(head))

        def swap(m: re.Match) -> str:
            nonlocal n
            after = text[m.end():]
            before = text[:m.start()]
            if (after[:1] and "ា" <= after[0] <= "៓") or after.startswith(_LONGER) \
                    or before.endswith("្"):
                return m.group(0)                       # part of a longer word
            n += 1
            return head + desc

        text = wrong.sub(swap, text)
    return text, n


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    bad: list[str] = []

    def expect(name: str, got, want) -> None:
        ok = got == want
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f"   got {got!r} want {want!r}"))
        if not ok:
            bad.append(name)

    expect("班长 in Chinese order put right",
           fix("班长681，沈知夏", "ថ្នាក់ប្រធានបាន៦៨១ សិន ជឺសៀ។"),
           ("ប្រធានថ្នាក់បាន៦៨១ សិន ជឺសៀ។", 1))
    expect("already in Khmer order: unchanged",
           fix("班长681", "ប្រធានថ្នាក់បាន៦៨១"), ("ប្រធានថ្នាក់បាន៦៨១", 0))
    expect("not said in the original: unchanged", fix("你好", "ថ្នាក់ប្រធាន"), ("ថ្នាក់ប្រធាន", 0))
    expect("副班长 is not taken for 班长",
           fix("副班长来了", "ថ្នាក់អនុប្រធានមកហើយ"), ("អនុប្រធានថ្នាក់មកហើយ", 1))
    expect("the start of a longer word is left",
           fix("校长", "សាលានាយកដ្ឋាន"), ("សាលានាយកដ្ឋាន", 0))
    expect("a space between is fine", fix("总分684", "សរុប ពិន្ទុ ៦៨៤"), ("ពិន្ទុសរុប ៦៨៤", 1))
    expect("the rule names the formula and the example",
           "ប្រធានថ្នាក់ (never ថ្នាក់ប្រធាន)" in rule("zh") and "noun FIRST" in rule("en"), True)
    expect("every compound is noun + describer, no duplicates",
           len({zh for zh, _h, _d in COMPOUNDS}) == len(COMPOUNDS), True)
    print("PASS" if not bad else f"FAILED: {bad}")
    raise SystemExit(1 if bad else 0)
