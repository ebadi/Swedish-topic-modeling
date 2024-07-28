import re
import sys
# regex are taken from https://github.com/peterdalle/markivet/


def miniMarkivet_parser(filename):
    file = open(filename, 'r', encoding='utf-8')
    Lines = file.readlines()
    Lines = Lines + ["========="]
    empty_article = {"text": '', "title": '', "date": '', "section": '', "page": '',
                     "published": '', "url": '', "newspaper": '', "bildtext": '', "copyright": ''}
    article = empty_article.copy()
    for line in Lines:
        metadata = False
        if line.startswith("> "):
            article = empty_article.copy()
            metadata = False
            continue
        if line.startswith("========="):
            yield article
            article = empty_article.copy()
            metadata = False
            continue
        date = re.search(
            r"\d{4}-\d{2}-\d{2}\s*(?:\d{2}:\d{2}(?::\d{2})?)?", line)
        if date:
            article["date"] = date.group().split(" ")[0].strip()
            metadata = True

        copyright_pattern = r'©(.*?)<br>'
        copyright = re.search(copyright_pattern, line)
        if copyright:
            article["copyright"] = copyright.group()
            line = re.sub(copyright_pattern, '', line, 1)

        # section = re.search(r"([A-ZÅÄÖ0-9]){2}\.", line)
        # if section:
        #    article["section"] = line.split(",")
        #    metadata= True

        newspaper = re.search(r"(\d{4}-\d{1,2})", line)
        if newspaper and len(article["newspaper"]) < 5:
            article["newspaper"] = line.split(",")[0].strip()
            metadata = True

        if "http://ret.nu/" in line or "https://ret.nu/" in line:
            article["url"] = re.search(
                "(?P<url>https?://[^\s]+)", line).group("url")
            metadata = True

        if line.startswith("Publicerat i") or line.startswith("Publicerat på") or line.startswith("Sänt i"):
            article["published"] = line.strip()
            metadata = True

        if line.startswith("Sida"):
            article["page"] = line.strip()
            metadata = True

        badword = "Alla artiklar är skyddade"
        if badword in line:
            index = line.find(badword)
            if index != -1:
                line = line[:index]

        if metadata is False:
            article["text"] = (article["text"] + line)


def extract_title(text):
    newtext = ""
    title = ""
    parts = text.split("\n")
    for part in parts:
        if len(title) < 10 and len(part) > 10:
            title = part
        else:
            newtext = newtext + "\n" + part
    return title, newtext


def load_markivet_file(filename):
    for article in miniMarkivet_parser(filename):
        article["title"], article["text"] = extract_title(article["text"])
        article["text"] = article["text"].replace("\n\n\n", "\n\n")
        article["text"] = article["text"].replace("\n\n\n", "\n\n")
        article["text"] = article["text"].replace("\n\n\n", "\n\n")
        article["text"] = article["text"].replace("\n\n\n", "\n\n")
        parts = article["text"].split("\n\n")

        article["text"] = ""
        article["bildtext"] = ""

        for part in parts:
            if part.startswith("Bildtext:"):
                article["bildtext"] = part
            else:
                article["text"] = article["text"] + "\n\n" + part
        article["text"] = article["text"].strip()
        yield article


def miniArkivet(directory="."):
    if directory == 'dummy':
        import pandas as pd
        dfnew = pd.read_csv('imoore-60k-stack-overflow-questions-with-quality-rate/valid.csv', encoding='ISO-8859-1') # utf8
        # print(list(dfnew.columns))
        dfnew['title'] = dfnew['Title']
        dfnew['text'] = dfnew['Body']
        def onlydate(txt) : return txt.split(' ')[0]
        dfnew['date'] = dfnew['CreationDate'].apply(onlydate)
        dfnew['newspaper'] = "Hagens Knuheter"
        dfnew['copyright'] ="©Sonnier<br>"
        def urlmaker(txt) : return  "https://stackoverflow.com/questions/" + str(txt)
        dfnew['url'] = dfnew['Id'].apply(urlmaker)
        dfnew['page'] = 1
        dfnew['Published'] = 'stackoverflow '
        dfnew['bildtext'] = 'bild text'
        dfnew = dfnew.drop(columns=['Title', 'Id', 'Body', 'CreationDate', 'Y', 'Tags'])
        return dfnew
    import pandas as pd
    import os
    df = pd.DataFrame(columns=['title', 'bildtext', 'date',
                      'published', 'url', 'newspaper', 'page', 'copyright', 'text'])
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            Markivet = load_markivet_file(os.path.join(directory, filename))
            for article in Markivet:
                df.loc[len(df)] = [article["title"], article["bildtext"],  article["date"], article["published"],
                                   article["url"], article["newspaper"], article["page"], article["copyright"], article["text"]]
            continue
        else:
            continue
    return df

if __name__ == '__main__':
    if sys.argv[1] == 'list':
        for record in load_markivet_file(sys.argv[2]):
            print("TITLE:", record["title"])
            print("BILDTEXT:", record["bildtext"])
            print("DATE:", record["date"])
            print("PUBLISHED:", record["published"])
            print("URL:", record["url"])
            print("NEWSPAPER:", record["newspaper"])
            print("PAGE:", record["page"])
            print("COPYRIGHT:", record["copyright"])
            print("TEXT:", record["text"])
            print("/////////////////////////////////////")
    elif sys.argv[1] == 'pandas':
        df = miniArkivet("tests")
        print(df.describe())
        print(df)
