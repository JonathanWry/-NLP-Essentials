import re

def chronicles_of_narnia(path):
    books = {}

    cur_book = None
    chapters = []
    cur_chapter = None
    token_count = 0

    waiting_chapter_num = None   
    skip_title_line = False  

    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

    def roman_to_int(r):
        r = r.upper()
        total = 0
        prev = 0
        for ch in reversed(r):
            if ch not in roman_map:
                return None
            val = roman_map[ch]
            if val < prev:
                total -= val
            else:
                total += val
                prev = val
        return total

    with open(path, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            toks = line.split()

            year_idx = -1
            year = None
            for i, t in enumerate(toks):
                if t.isdigit() and len(t) == 4:
                    year_idx = i
                    year = int(t)
                    break

            is_book = False
            if year_idx >= 0:
                if year_idx - 1 >= 0 and year_idx + 1 < len(toks) and toks[year_idx - 1] == '(' and toks[year_idx + 1] == ')':
                    title = ' '.join(toks[:year_idx - 1]).strip()
                    if title:
                        is_book = True

            if is_book:
                # flush previous chapter
                if cur_chapter is not None:
                    cur_chapter['token_count'] = token_count
                    chapters.append(cur_chapter)
                    cur_chapter = None
                    token_count = 0

                if cur_book is not None:
                    chapters.sort(key=lambda c: c['number'])
                    books[cur_book]['chapters'] = chapters

                cur_book = title
                books[cur_book] = {'title': title, 'year': year, 'chapters': []}
                chapters = []

                waiting_chapter_num = None
                skip_title_line = False
                continue

            if len(toks) >= 2 and toks[0] in {'CHAPTER', 'Chapter', 'chapter'} and cur_book is not None:
                num = roman_to_int(toks[1])
                if num is not None:
                    # flush previous chapter
                    if cur_chapter is not None:
                        cur_chapter['token_count'] = token_count
                        chapters.append(cur_chapter)
                        cur_chapter = None
                        token_count = 0

                    waiting_chapter_num = num
                    skip_title_line = True
                    continue

            if skip_title_line and waiting_chapter_num is not None and cur_book is not None:
                chap_title = ' '.join(toks).strip()
                cur_chapter = {'number': waiting_chapter_num, 'title': chap_title, 'token_count': 0}
                token_count = 0
                waiting_chapter_num = None
                skip_title_line = False
                continue

            if cur_book is not None and cur_chapter is not None:
                token_count += len(toks)

    if cur_chapter is not None:
        cur_chapter['token_count'] = token_count
        chapters.append(cur_chapter)

    if cur_book is not None:
        chapters.sort(key=lambda c: c['number'])
        books[cur_book]['chapters'] = chapters

    return books

def regular_expressions(text):
    s = text.strip()

    email_re = re.compile(
        r'^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?@'
        r'[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?'
        r'\.(?:com|org|edu|gov)$'
    )

    date_re = re.compile(r'^(?P<y>\d{2}|\d{4})(?P<sep>[/-])(?P<m>\d{1,2})(?P=sep)(?P<d>\d{1,2})$')

    url_re = re.compile(
        r'^https?://'
        r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?'
        r'(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+$'
    )

    last = r'[A-Z][a-z]+(?:[- ][A-Z][a-z]+)*'
    cite_re = re.compile(
        r'^' + last + r'(?:'
        r',\s*\d{4}'
        r'|\s+and\s+' + last + r',\s*\d{4}'
        r'|\s+et\s+al\.,\s*\d{4}'
        r')$'
    )

    if email_re.fullmatch(s):
        return 'email'

    m = date_re.fullmatch(s)
    if m:
        y_raw = m.group('y')
        month = int(m.group('m'))
        day = int(m.group('d'))

        if len(y_raw) == 4:
            year = int(y_raw)
        else:
            yy = int(y_raw)
            year = 1900 + yy if 51 <= yy <= 99 else 2000 + yy  # 51-99 => 1951-1999, else 2000-2050

        def leap(y):
            return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

        if 1951 <= year <= 2050 and 1 <= month <= 12 and day >= 1:
            dim = {
                1: 31, 2: 29 if leap(year) else 28, 3: 31, 4: 30,
                5: 31, 6: 30, 7: 31, 8: 31,
                9: 30, 10: 31, 11: 30, 12: 31
            }
            if day <= dim[month]:
                return 'date'

    if url_re.fullmatch(s):
        return 'url'

    if cite_re.fullmatch(s):
        year = int(re.findall(r'\d{4}', s)[-1])
        if 1900 <= year <= 2024:
            return 'cite'

    return None
