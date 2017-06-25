# -*-coding:utf-8-*-
import re
import urllib.error
import urllib.parse
import urllib.request


def seg(text):
    if not isinstance(text, str):
        try:
            text = text.decode('utf-8')
        except:
            raise UnicodeError('Input encoding should be UTF8 or UNICODE')
    try:
        text = text.encode('cp950')
    except:
        raise Exception(
            'CKIP Segmentator only accepts characters encoded in CP950; however, it seems that there are some characters which cannot be encoded in CP950.')
    url_tar = 'http://sunlight.iis.sinica.edu.tw/cgi-bin/text.cgi'

    # text = hack(text)

    opener = urllib.request.build_opener()
    postdata = urllib.parse.urlencode({
        'query': text,
        'Submit': '送出'.encode('cp950')
    }).encode('cp950')

    res = opener.open(url_tar, postdata).read()
    pat = re.compile(b"URL=\'/uwextract/pool/(\d*?).html\'")
    num = pat.search(res).group(1)

    url_fin = 'http://sunlight.iis.sinica.edu.tw/uwextract/show.php?id=%s&type=tag' % int(num)
    seg = urllib.request.urlopen(url_fin).read()
    break_sign = b'-' * 130

    seg_pat = re.compile(b'<pre>(.*?)</pre>', re.DOTALL)
    seg_clean = seg_pat.search(seg).group(1)
    seg_clean = seg_clean.replace(break_sign, b'')
    seg_clean = seg_clean.decode('cp950', 'ignore')
    seg_clean = seg_clean.strip('\n')
    fs = '\u3000'  # fullwidth space
    seg_clean = seg_clean.strip(fs)
    seg_fin = seg_clean.split(fs)
    seg_fin_pat = re.compile('(.*?)\((\w*?)\)')
    con = []
    for i in seg_fin:
        o = seg_fin_pat.search(i)
        if o is not None:  # need to find out why None appears!
            con.append((o.group(1), o.group(2)))
    output = Segres(con)
    return output


def num_patch(string):
    num_h = [chr(i) for i in range(48, 58)]
    num_f = [chr(i) for i in range(65296, 65306)]
    num_patch = dict(list(zip(num_f, num_h)))
    output = ''
    for i in string:
        if i in list(num_patch.keys()):
            output += num_patch[i]
        else:
            output += i
    return output


class Segres(object):
    def __init__(self, object):
        raw = object
        output = ''
        for word, pos in raw:
            output += '%s/%s ' % (word, pos)
        output = output.replace('__n__/FW', '\n')
        output = output.replace('__n__', '\n')
        output = output.replace('__<__', '<')
        output = output.replace('__>__', '>')
        output = output.strip()
        output = output.split(' ')
        res = []
        for i in output:
            tmp = i.partition('\n')
            for x in tmp:
                if x != '':
                    res.append(x)
        fin = []
        for i in res:
            if i == '\n':
                word = i
                pos = 'LINEBREAK'
            else:
                pat = re.search('(.*)/(\w+)$', i)
                # print i, pat
                try:  # need to check
                    word = pat.group(1)
                except:
                    word = i
                word = num_patch(word)
                try:
                    pos = pat.group(2)
                except:
                    pos = 'None'
            fin.append((word, pos))

        self.raw = fin

    def text(self, mode='plain'):
        output = ''
        for word, pos in self.raw:
            if pos == 'LINEBREAK':
                output += word
            else:
                if mode == 'plain':
                    output += '%s/%s ' % (word, pos)
                elif mode == 'html':
                    output += '%s<span>/%s</span> ' % (word, pos)
                else:
                    raise ValueError('Mode name error: %s' % mode)
        return output

    def nopos(self, mode='string'):
        output = [word for word, pos in self.raw]
        if mode == 'string':
            output = ' '.join(output)
            output = output.replace(' \n ', '\n')
        elif mode == 'list':
            pass
        else:
            raise ValueError('Mode name error: %s' % mode)
        return output
