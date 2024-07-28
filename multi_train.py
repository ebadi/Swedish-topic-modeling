import itertools    
from umap import UMAP
from hdbscan import HDBSCAN

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction


def all_options():

    embedding_model_0 = ("paraphrase-multilingual", SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"))
    embedding_model_1 = ("KBLab", SentenceTransformer("KBLab/sentence-bert-swedish-cased"))

    umap_model_0 = ("umap_n5", UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'))
    umap_model_1 = ("umap_n10", UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine'))

    hdbscan_model_0 = ("HDBSCAN_c15", HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True))
    hdbscan_model_1 = ("HDBSCAN_c30", HDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom', prediction_data=True))

    SEstopwords = ["aderton","adertonde","adjö","aldrig","alla","allas","allt","alltid","alltså","andra","andras","annan","annat","artonde","artonn","att","av","bakom","bara","behöva","behövas","behövde","behövt","beslut","beslutat","beslutit","bland","blev","bli","blir","blivit","bort","borta","bra","bäst","bättre","båda","bådas","dag","dagar","dagarna","dagen","de","del","delen","dem","den","denna","deras","dess","dessa","det","detta","dig","din","dina","dit","ditt","dock","dom","du","där","därför","då","e","efter","eftersom","ej","elfte","eller","elva","emot","en","enkel","enkelt","enkla","enligt","ens","er","era","ers","ert","ett","ettusen","fanns","fem","femte","femtio","femtionde","femton","femtonde","fick","fin","finnas","finns","fjorton","fjortonde","fjärde","fler","flera","flesta","fram","framför","från","fyra","fyrtio","fyrtionde","få","får","fått","följande","för","före","förlåt","förra","första","genast","genom","gick","gjorde","gjort","god","goda","godare","godast","gott","gälla","gäller","gällt","gärna","gå","går","gått","gör","göra","ha","hade","haft","han","hans","har","heller","hellre","helst","helt","henne","hennes","hit","hon","honom","hundra","hundraen","hundraett","hur","här","hög","höger","högre","högst","i","ibland","icke","idag","igen","igår","imorgon","in","inför","inga","ingen","ingenting","inget","innan","inne","inom","inte","inuti","ja","jag","jo","ju","just","jämfört","kan","kanske","knappast","kom","komma","kommer","kommit","kr","kunde","kunna","kunnat","kvar","legat","ligga","ligger","lika","likställd","likställda","lilla","lite","liten","litet","länge","längre","längst","lätt","lättare","lättast","långsam","långsammare","långsammast","långsamt","långt","låt","man","med","mej","mellan","men","mer","mera","mest","mig","min","mina","mindre","minst","mitt","mittemot","mot","mycket","många","måste","möjlig","möjligen","möjligt","möjligtvis","ned","nederst","nedersta","nedre","nej","ner","ni","nio","nionde","nittio","nittionde","nitton","nittonde","nog","noll","nr","nu","nummer","när","nästa","någon","någonting","något","några","nån","nånting","nåt","nödvändig","nödvändiga","nödvändigt","nödvändigtvis","och","också","ofta","oftast","olika","olikt","om","oss","på","rakt","redan","rätt","sa","sade","sagt","samma","sedan","senare","senast","sent","sex","sextio","sextionde","sexton","sextonde","sig","sin","sina","sist","sista","siste","sitt","sitta","sju","sjunde","sjuttio","sjuttionde","sjutton","sjuttonde","själv","sjätte","ska","skall","skulle","slutligen","små","smått","snart","som","stor","stora","stort","större","störst","säga","säger","sämre","sämst","så","sådan","sådana","sådant","ta","tack","tar","tidig","tidigare","tidigast","tidigt","till","tills","tillsammans","tio","tionde","tjugo","tjugoen","tjugoett","tjugonde","tjugotre","tjugotvå","tjungo","tolfte","tolv","tre","tredje","trettio","trettionde","tretton","trettonde","två","tvåhundra","under","upp","ur","ursäkt","ut","utan","utanför","ute","va","vad","var","vara","varför","varifrån","varit","varje","varken","vars","varsågod","vart","vem","vems","verkligen","vi","vid","vidare","viktig","viktigare","viktigast","viktigt","vilka","vilkas","vilken","vilket","vill","väl","vänster","vänstra","värre","vår","våra","vårt","än","ännu","är","även","åt","åtminstone","åtta","åttio","åttionde","åttonde","över","övermorgon","överst","övre"]
    vectorizer_model_0 = ("SE_stopword", CountVectorizer(stop_words=SEstopwords))

    ctfidf_model_0 =  ("TfidfTransformer", ClassTfidfTransformer())

    representation_model_0 = ("KeyBERTInspired", KeyBERTInspired())

    options = [ [embedding_model_0, embedding_model_1], [umap_model_0, umap_model_1], [hdbscan_model_0, hdbscan_model_1] ,[vectorizer_model_0], [ctfidf_model_0], [representation_model_0]]

    return options

def build_all_models(docs):
  for (id, option) in  enumerate(itertools.product(*all_options())):
    (embedding_name, embedding_model),(umap_name, umap_model),(hdbscan_name, hdbscan_model) ,(vectorizer_name, vectorizer_model), (ctfidf_name, ctfidf_model) , (representation_name, representation_model) = option
    config_name = embedding_name + "_" + umap_name + "_" + hdbscan_name + "_" +  vectorizer_name + "_" + ctfidf_name + "_" + representation_name
    print(config_name)
    topic_model = BERTopic(
      embedding_model=embedding_model,
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      vectorizer_model=vectorizer_model,
      ctfidf_model=ctfidf_model,
      representation_model=representation_model
    )
    topic_model.fit(docs)
    topic_model.save("results/" + config_name + ".pickle", serialization="pickle")
