import unittest
from word_embedding import *

class MyTestCase(unittest.TestCase):
    #测试函数fastTextWordEmbedding
    def test_fastTextWordEmbedding(self):
        test_a=fastTextWordEmbedding()
        test_b=test_a
        a,b=False,False
        if isinstance(test_a, Vectors):
            a=True
        if isinstance(test_b, Vectors):
            b=True
        self.assertTrue(a)
        self.assertTrue(b)
    def test_wordEncoding1(self):
        documents=['a b a a a a b b b b c c d e ']
        a = False

        ans=wordEncoding(documents,NumWords=5,Order=None)
        true_ans=['a','b','c','d','e']
        if isinstance(ans, dictionary):
            a=True
        self.assertTrue(a)
        temp = []
        for value in ans.token2id:
            temp += [value]
        self.assertEqual(temp, true_ans)

    def test_wordEncoding2(self):
        documents1='test_wordEncoding.txt'
        documents2 = [
             "fairest creatures desire increase thereby beautys rose might never die riper time decease tender heir might bear memory thou contracted thine own bright eyes feedst thy lights flame selfsubstantial fuel making famine abundance lies thy self thy foe thy sweet self cruel thou art worlds fresh ornament herald gaudy spring thine own bud buriest thy content tender churl makst waste niggarding pity world else glutton eat worlds due grave thee",
            "forty winters shall besiege thy brow dig deep trenches thy beautys field thy youths proud livery gazed tatterd weed small worth held asked thy beauty lies treasure thy lusty days say thine own deep sunken eyes alleating shame thriftless praise praise deservd thy beautys thou couldst answer fair child mine shall sum count make old excuse proving beauty succession thine new made thou art old thy blood warm thou feelst cold",
            "look thy glass tell face thou viewest time face form another whose fresh repair thou renewest thou dost beguile world unbless mother fair whose uneard womb disdains tillage thy husbandry fond tomb selflove stop posterity thou art thy mothers glass thee calls back lovely april prime thou windows thine age shalt despite wrinkles thy golden time thou live rememberd die single thine image dies thee",
            "unthrifty loveliness why dost thou spend upon thy self thy beautys legacy natures bequest gives nothing doth lend frank lends free beauteous niggard why dost thou abuse bounteous largess thee give profitless usurer why dost thou great sum sums yet canst live traffic thy self alone thou thy self thy sweet self dost deceive nature calls thee gone acceptable audit canst thou leave thy unused beauty tombed thee lives th executor",
            "hours gentle work frame lovely gaze every eye doth dwell play tyrants same unfair fairly doth excel neverresting time leads summer hideous winter confounds sap checked frost lusty leaves quite gone beauty oersnowed bareness every summers distillation left liquid prisoner pent walls glass beautys effect beauty bereft nor nor remembrance flowers distilld though winter meet leese show substance still lives sweet",
            "let winters ragged hand deface thee thy summer ere thou distilld make sweet vial treasure thou place beautys treasure ere selfkilld forbidden usury happies pay willing loan thats thy self breed another thee ten times happier ten ten times thy self happier thou art ten thine ten times refigurd thee death thou shouldst depart leaving thee living posterity selfwilld thou art fair deaths conquest make worms thine heir",
            "lo orient gracious light lifts up burning head eye doth homage newappearing sight serving looks sacred majesty climbd steepup heavenly hill resembling strong youth middle age yet mortal looks adore beauty still attending golden pilgrimage highmost pitch weary car like feeble age reeleth day eyes fore duteous converted low tract look another way thou thyself outgoing thy noon unlookd diest unless thou get son",
            "music hear why hearst thou music sadly sweets sweets war joy delights joy why lovst thou thou receivst gladly else receivst pleasure thine annoy true concord welltuned sounds unions married offend thine ear sweetly chide thee confounds singleness parts thou shouldst bear mark string sweet husband another strikes mutual ordering resembling sire child happy mother pleasing note sing whose speechless song many seeming sings thee thou single wilt prove none",
            "fear wet widows eye thou consumst thy self single life ah thou issueless shalt hap die world wail thee like makeless wife world thy widow still weep thou form thee hast left behind every private widow well keep childrens eyes husbands shape mind look unthrift world doth spend shifts place still world enjoys beautys waste hath world end kept unused user destroys love toward others bosom sits murdrous shame commits",
            "shame deny thou bearst love thy self art unprovident grant thou wilt thou art belovd many thou none lovst evident thou art possessd murderous hate gainst thy self thou stickst conspire seeking beauteous roof ruinate repair thy chief desire o change thy thought change mind shall hate fairer lodgd gentle love thy presence gracious kind thyself least kindhearted prove make thee another self love beauty still live thine thee"
        ]
        ans1=wordEncoding(documents1,NumWords=1)
        ans2=wordEncoding(documents2,Order=None)
        a,b= False,False
        if isinstance(ans1, dictionary):
            a = True
        if isinstance(ans2, dictionary):
            b = True
        true_ans=['abundance', 'art', 'bear', 'beautys', 'bright', 'bud', 'buriest', 'churl', 'content',
                  'contracted', 'creatures', 'cruel', 'decease', 'desire', 'die', 'due', 'eat', 'else', 'eyes', 'fairest',
                  'famine', 'feedst', 'flame', 'foe', 'fresh', 'fuel', 'gaudy', 'glutton', 'grave', 'heir', 'herald',
                  'increase', 'lies', 'lights', 'making', 'makst', 'memory', 'might', 'never', 'niggarding', 'ornament',
                  'own', 'pity', 'riper', 'rose', 'self', 'selfsubstantial', 'spring', 'sweet', 'tender', 'thee', 'thereby',
                  'thine', 'thou', 'thy', 'time', 'waste', 'world', 'worlds', 'alleating', 'answer', 'asked', 'beauty',
                  'besiege', 'blood', 'brow', 'child', 'cold', 'couldst', 'count', 'days', 'deep', 'deservd', 'dig',
                  'excuse', 'fair', 'feelst', 'field', 'forty', 'gazed', 'held', 'livery', 'lusty', 'made', 'make', 'mine',
                  'new', 'old', 'praise', 'proud', 'proving', 'say', 'shall', 'shame', 'small', 'succession', 'sum', 'sunken',
                  'tatterd', 'thriftless', 'treasure', 'trenches', 'warm', 'weed', 'winters', 'worth', 'youths', 'age',
                  'another', 'april', 'back', 'beguile', 'calls', 'despite', 'dies', 'disdains', 'dost', 'face', 'fond',
                  'form', 'glass', 'golden', 'husbandry', 'image', 'live', 'look', 'lovely', 'mother', 'mothers', 'posterity',
                  'prime', 'rememberd', 'renewest', 'repair', 'selflove', 'shalt', 'single', 'stop', 'tell', 'tillage',
                  'tomb', 'unbless', 'uneard', 'viewest', 'whose', 'windows', 'womb', 'wrinkles', 'abuse', 'acceptable',
                  'alone', 'audit', 'beauteous', 'bequest', 'bounteous', 'canst', 'deceive', 'doth', 'executor', 'frank',
                  'free', 'give', 'gives', 'gone', 'great', 'largess', 'leave', 'legacy', 'lend', 'lends', 'lives',
                  'loveliness', 'nature', 'natures', 'niggard', 'nothing', 'profitless', 'spend', 'sums', 'th', 'tombed',
                  'traffic', 'unthrifty', 'unused', 'upon', 'usurer', 'why', 'yet', 'bareness', 'bereft', 'checked',
                  'confounds', 'distillation', 'distilld', 'dwell', 'effect', 'every', 'excel', 'eye', 'fairly', 'flowers',
                  'frame', 'frost', 'gaze', 'gentle', 'hideous', 'hours', 'leads', 'leaves', 'leese', 'left', 'liquid',
                  'meet', 'neverresting', 'nor', 'oersnowed', 'pent', 'play', 'prisoner', 'quite', 'remembrance', 'same',
                  'sap', 'show', 'still', 'substance', 'summer', 'summers', 'though', 'tyrants', 'unfair', 'walls', 'winter',
                  'work', 'breed', 'conquest', 'death', 'deaths', 'deface', 'depart', 'ere', 'forbidden', 'hand', 'happier',
                  'happies', 'leaving', 'let', 'living', 'loan', 'pay', 'place', 'ragged', 'refigurd', 'selfkilld',
                  'selfwilld', 'shouldst', 'ten', 'thats', 'times', 'usury', 'vial', 'willing', 'worms', 'adore', 'attending',
                  'burning', 'car', 'climbd', 'converted', 'day', 'diest', 'duteous', 'feeble', 'fore', 'get', 'gracious',
                  'head', 'heavenly', 'highmost', 'hill', 'homage', 'lifts', 'light', 'like', 'lo', 'looks', 'low', 'majesty',
                  'middle', 'mortal', 'newappearing', 'noon', 'orient', 'outgoing', 'pilgrimage', 'pitch', 'reeleth',
                  'resembling', 'sacred', 'serving', 'sight', 'son', 'steepup', 'strong', 'thyself', 'tract', 'unless',
                  'unlookd', 'up', 'way', 'weary', 'youth', 'annoy', 'chide', 'concord', 'delights', 'ear', 'gladly', 'happy',
                  'hear', 'hearst', 'husband', 'joy', 'lovst', 'many', 'mark', 'married', 'music', 'mutual', 'none', 'note',
                  'offend', 'ordering', 'parts', 'pleasing', 'pleasure', 'prove', 'receivst', 'sadly', 'seeming', 'sing',
                  'singleness', 'sings', 'sire', 'song', 'sounds', 'speechless', 'strikes', 'string', 'sweetly', 'sweets',
                  'true', 'unions', 'war', 'welltuned', 'wilt', 'ah', 'behind', 'bosom', 'childrens', 'commits', 'consumst',
                  'destroys', 'end', 'enjoys', 'fear', 'hap', 'hast', 'hath', 'husbands', 'issueless', 'keep', 'kept', 'life',
                  'love', 'makeless', 'mind', 'murdrous', 'others', 'private', 'shape', 'shifts', 'sits', 'toward', 'unthrift',
                  'user', 'wail', 'weep', 'well', 'wet', 'widow', 'widows', 'wife', 'bearst', 'belovd', 'change', 'chief',
                  'conspire', 'deny', 'evident', 'fairer', 'gainst', 'grant', 'hate', 'kind', 'kindhearted', 'least', 'lodgd',
                  'murderous', 'possessd', 'presence', 'roof', 'ruinate', 'seeking', 'stickst', 'thought', 'unprovident']
        self.assertTrue(a)
        self.assertTrue(b)
        temp=[]
        for value in ans1.token2id:
            temp += [value]
        self.assertEqual(temp,true_ans)
        self.assertEqual(len(ans1),len(true_ans))
        self.assertEqual(len(ans2),418)

    def test_doc2sequence1(self):

        documents = 'test_wordEncoding.txt'
        wv1=fastTextWordEmbedding()
        wv2=wordEncoding(documents)

        new_doc=['never the of summer king','rule eye the homo self']
        new_doc1='eye summer of'
        ans1=[[0, 2, 953, 1292, 2466], [0, 624, 1235]]
        ans2=[[38, 226], [45, 198]]
        ans3=[[2, 2466]]
        new_vec1=doc2sequence(wv1,new_doc)
        new_vec2=doc2sequence(wv2,new_doc)
        new_vec3 = doc2sequence(wv1, new_doc1)
        self.assertEqual(new_vec1, ans1)
        self.assertEqual(new_vec2, ans2)
        self.assertEqual(new_vec3, ans3)

    def test_doc2sequence2(self):
        documents = 'test_wordEncoding.txt'
        wv = wordEncoding(documents)
        new_doc = ['head eye music sadly frame work', 'poi poi poi poi','head head head single life']
        new_vec1 = doc2sequence(wv, new_doc)
        ans1=[[198, 201, 233, 276, 327, 338], [], [136, 276, 373]]
        new_vec2 = doc2sequence(wv, new_doc, PaddingDirection='left',Length='longest')
        ans2=[[198, 201, 233, 276, 327, 338], [0, 0, 0, 0, 0, 0], [0, 0, 0, 136, 276, 373]]
        new_vec3 = doc2sequence(wv, new_doc, PaddingDirection='right', PaddingValue=66)
        ans3=[[198, 201, 233, 276, 327, 338], [66, 66, 66, 66, 66, 66], [136, 276, 373, 66, 66, 66]]
        new_vec4 = doc2sequence(wv, new_doc, PaddingDirection='left',Length='shortest')
        ans4=[[198, 201, 233, 276, 327, 338], [], [136, 276, 373]]
        new_vec5 = doc2sequence(wv, new_doc, PaddingDirection='right', PaddingValue=100,Length=10)
        ans=[[198, 201, 233, 276, 327, 338, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100, 100, 100, 100], [136, 276, 373, 100, 100, 100, 100, 100, 100, 100]]

    def test_readWordEmbedding(self):
        wv=readWordEmbedding('my_model1.bin')
        a, b = False, False
        if isinstance(wv, Vectors):
            a = True
        ans_key=['the', 'to', 'of', 'in', 'and', 'he', 'is', 'for', 'on', 'said', 'that', 'has', 'says', 'was', 'have', 'it', 'be', 'are', 'with',
                 'will', 'at', 'mr', 'from', 'by', 'we', 'been', 'as', 'an', 'not', 'his', 'but', 'they', 'after', 'were', 'had', 'there', 'new',
                 'this', 'australian', 'australia', 'who', 'people', 'palestinian', 'their', 'government', 'two', 'up', 'south', 'us', 'which',
                 'year', 'one', 'about', 'out', 'if', 'also', 'more', 'when', 'its', 'into', 'would', 'first', 'against', 'last', 'israeli', 'minister',
                 'arafat', 'all', 'over', 'afghanistan', 'three', 'united', 'no', 'world', 'police', 'or', 'than', 'attacks', 'before', 'fire', 'day', 'security', 'some', 'states', 'you', 'could', 'them', 'say', 'today', 'now', 'told', 'time', 'any', 'very', 'laden', 'bin', 'just', 'can', 'what', 'president', 'sydney', 'company', 'still', 'four', 'man', 'killed', 'taliban', 'al', 'forces', 'our', 'around', 'west', 'being', 'days', 'old', 'other', 'officials', 'where', 'so', 'test', 'qaeda', 'israel', 'next', 'per', 'general', 'think', 'federal', 'she', 'force', 'cent', 'workers', 'leader', 'yesterday', 'under', 'hamas', 'take', 'state', 'him', 'those', 'bank', 'years', 'back', 'meeting', 'suicide', 'made', 'morning', 'action', 'down', 'commission', 're', 'pakistan', 'international', 'attack', 'centre', 'afghan', 'group', 'city', 'well', 'through', 'military', 'members', 'while', 'number', 'five', 'called', 'local', 'area', 'qantas', 'gaza', 'week', 'union', 'national', 'since', 'hours', 'wales', 'september', 'including', 'another', 'east', 'report', 'night', 'off', 'north', 'should', 'six', 'staff', 'between', 'these', 'get', 'second', 'earlier', 'war', 'go', 'islamic', 'further', 'defence', 'end', 'months', 'do', 'because', 'authority', 'foreign', 'going', 'power', 'work', 'areas', 'near', 'team', 'sharon', 'during', 'died', 'month', 'only', 'many', 'india', 'way', 'eight', 'know', 'metres', 'match', 'good', 'make', 've', 'melbourne', 'northern', 'spokesman', 'claims', 'former', 'left', 'air', 'most', 'support', 'osama', 'peace', 'like', 'authorities', 'prime', 'given', 'am', 'ago', 'saying', 'set', 'expected', 'tora', 'put', 'bora', 'looking', 'come', 'place', 'militants', 'fighters', 'several', 'children', 'meanwhile', 'injured', 'christmas', 'groups', 'indian', 'found', 'unions', 'arrested', 'royal', 'river', 'troops', 'child', 'africa', 'talks', 'official', 'whether', 'interim', 'reports', 'then', 'hospital', 'terrorist', 'yasser', 'part', 'industrial', 'don', 'start', 'how', 'statement', 'leaders', 'third', 'early', 'senior', 'terrorism', 'economy', 'mountains', 'weather', 'hit', 'million', 'believe', 'trying', 'family', 'both', 'john', 'army', 'pay', 'court', 'radio', 'public', 'dr', 'asked', 'control', 'lead', 'pressure', 'best', 'long', 'adelaide', 'chief', 'following', 'however', 'agreement', 'help', 'few', 'house', 'play', 'labor', 'australians', 'arrest', 'better', 'want', 'does', 'firefighters', 'high', 'need', 'close', 'service', 'community', 'taken', 'confirmed', 'queensland', 'services', 'overnight', 'process', 'opposition', 'williams', 'must', 'secretary', 'information', 'believed', 'came', 'detainees', 'won', 'governor', 'held', 'shot', 'damage', 'possible', 'her', 'hicks', 'nations', 'much', 'pentagon', 'peter', 'party', 'did', 'released', 'win', 'maintenance', 'took', 'brought', 'british', 'accused', 'safety', 'armed', 'kandahar', 'winds', 'despite', 'even', 'such', 'across', 'eastern', 'violence', 'weekend', 'return', 'conditions', 'without', 'building', 'lot', 'asylum', 'dead', 'cut', 'york', 'change', 'hill', 'director', 'council', 'airline', 'got', 'far', 'news', 'lee', 'waugh', 'trade', 'southern',
                 'crew', 'continuing', 'monday', 'captured', 'fires', 'see', 'race', 'economic', 'strong', 'call', 'anti', 'emergency', 'cricket', 'region', 'aircraft', 'palestinians', 'men', 'home', 'training', 'seekers', 'working', 'strip', 'country', 'american', 'david', 'board', 'role', 'here', 'december', 'too', 'fighting', 'plans', 'industry', 'george', 'charged', 'act', 'health', 'bush', 'received', 'key', 'alliance', 'rate', 'past', 'administration', 'bureau', 'used', 'head', 'offer', 'water', 'zinni', 'town', 'within', 'boat', 'decision', 'zealand', 'least', 'israelis', 'strikes', 'britain', 'line', 'department', 'soldiers', 'hih', 'station', 'issue', 'downer', 'leading', 'use', 'major', 'person', 'operations', 'hundreds', 'stop', 'final', 'parliament', 'known', 'captain', 'legal', 'large', 'airport', 'risk', 'your', 'may', 'later', 'give', 'ahead', 'officers', 'tomorrow', 'half', 'due', 'un', 'series', 'laws', 'able', 'interest', 'every', 'homes', 'taking', 'weapons', 'coast', 'behind', 'hollingworth', 'policy', 'network', 'western', 'kabul', 'pm', 'great', 'latest', 'late', 'plane', 'my', 'remain', 'might', 'right', 'shane', 'death', 'position', 'already', 'heard', 'deaths', 'forced', 'life', 'hard', 'failed', 'seen', 'continue', 'towards', 'along', 'jihad', 'side', 'timor', 'abuse', 'territory', 'special', 'others', 'guilty', 'campaign', 'bill', 'storm', 'same', 'flight', 'concerned', 'victory', 'cup', 'jobs', 'thought', 'event', 'me', 'point', 'really', 'washington', 'member', 'buildings', 'november', 'case', 'sunday', 'weeks', 'bombing', 'mcgrath', 'bowler', 'matter', 'th', 'innings', 'helicopters', 'bus', 'envoy', 'details', 'countries', 'likely', 'middle', 'canyoning', 'move', 'rates', 'situation', 'cabinet', 'again', 'capital', 'woomera', 'seven', 'bichel', 'warne', 'mark', 'launched', 'according', 'detention', 'enough', 'bombings', 'important', 'space', 'hour', 'office', 'wants', 'boy', 'human', 'adventure', 'perth', 'women', 'young', 'political', 'deal', 'asio', 'based', 'claimed', 'commonwealth', 'evidence', 'sure', 'justice', 'swiss', 'jail', 'reported', 'aedt', 'caught', 'own', 'movement', 'mission', 'african', 'disease', 'added', 'show', 'raids', 'blue', 'opened', 'run', 'money', 'forward', 'immediately', 'guides', 'rule', 'top', 'carried', 'freeze', 'warplanes', 'targets', 'cancer', 'ms', 'dispute', 'wicket', 'times', 'face', 'march', 'always', 'investigation', 'border', 'flying', 'full', 'allegations', 'sector', 'financial', 'although', 'growth', 'ground', 'burning', 'result', 'order', 'crash', 'planning', 'break', 'island', 'job', 'become', 'carrying', 'find', 'using', 'access', 'beat', 'prepared', 'each', 'certainly', 'banks', 'reached', 'proposed', 'probably', 'collapse', 'relations', 'inside', 'reserve', 'allow', 'tourists', 'militia', 'organisation', 'radical', 'militant', 'afp', 'road', 'wave', 'different', 'executive', 'energy', 'clear', 'serious', 'responsibility', 'hewitt', 'post', 'making', 'inquiry', 'sent', 'harrison', 'suspected', 'surrender', 'trees', 'management', 'thousands', 'thursday', 'needs', 'bowling', 'future', 'rejected', 'outside', 'opening', 'travel', 'kilometres', 'short', 'killing', 'ariel', 'quickly', 'howard', 'went', 'drop', 'bid', 'sex', 'himself', 'declared', 'comes', 'fight', 'lives', 'soon', 'ansett', 'immigration', 'caves', 'tried', 'stage', 'argentina', 'believes', 'senator', 'flights', 'calls', 'program', 'getting', 'until', 'alleged', 'sentence', 'circumstances', 'television', 'quite', 'annual', 'rural', 'wounded', 'try', 'open', 'attempt', 'white', 'terms', 'ruddock', 'among', 'offices', 'sea', 'currently', 'tanks', 'available', 'sources', 'ended', 'donald', 'anything', 'refused', 'look', 'figures', 'changes', 'means', 'measures', 'alexander', 'shortly', 'yacht', 'issues', 'form', 'july', 'victoria', 'running', 'bombers', 'agency', 'address', 'response', 'gunmen', 'karzai', 'meet', 'happened', 'directors', 'actually', 'jewish', 'yet', 'something', 'done', 'wickets', 'residents', 'returned', 'destroyed', 'brisbane', 'announced', 'accident', 'warned', 'understand', 'parts', 'rise', 'decided', 'ministers', 'witnesses', 'biggest', 'parties', 'nine', 'list', 'source', 'kallis', 'fighter', 'school', 'increase', 'rights', 'caused', 'allan', 'tour', 'highway', 'deputy',
                 'media', 'commissioner', 'arrived', 'facility', 'away', 'elected', 'though', 'advice', 'supporters', 'ses', 'total', 'pacific', 'cost', 'law', 'wage', 'fact', 'difficult', 'prevent', 'began', 'confidence', 'negotiations', 'blake', 'huge', 'rafter', 'beyond', 'rather', 'beginning',
                 'sir', 'glenn', 'business', 'afternoon', 'struck', 'intelligence', 'ensure', 'virus', 'period', 'big', 'tennis', 'islands', 'car', 'having', 'commander', 'appeared', 'step', 'powell', 'strike', 'crowd', 'whose', 'expect', 'ever', 'runs', 'commanders', 'suharto', 'airlines', 'market', 'nauru', 'll', 'less', 'attorney', 'expressed', 'gave', 'worst', 'involved', 'gives', 'followed', 'recession', 'musharraf', 'robert', 'field', 'friday', 'bomb', 'hobart', 'tragedy', 'terror', 'smoke', 'potential', 'plan', 'played', 'feel', 'reid', 'places', 'speaking', 'costs', 'anthony', 'entered', 'muslim', 'hearings', 'lockett', 'helicopter', 'front', 'storms', 'organisations', 'crisis', 'jenin', 'ambush', 'quarter', 'level', 'conflict', 'base', 'zimbabwe', 'endeavour', 'chance', 'macgill', 'longer', 'giving', 'employees', 'nearly', 'explosives', 'assistance', 'yes', 'geoff', 'changed', 'chairman', 'vote', 'criticism', 'heavy', 'main', 'complex', 'threat', 'game', 'brett', 'calling', 'jacques', 'lost', 'population', 'bring', 'study', 'saturday', 'russian', 'fast', 'round', 'indonesia', 'protect', 'sort', 'daryl', 'current', 'never', 'together', 'decide', 'survey', 'conference', 'massive', 'powers', 'boxing', 'anglican', 'labour', 'crean', 'americans', 'themselves', 'martin', 'finished', 'placed', 'denied', 'son', 'entitlements', 'ballot', 'cannot', 'significant', 'pakistani', 'debt', 'france', 'tasmania', 'trip', 'receive', 'flames', 'record', 'activity', 'bomber', 'closed', 'problems', 'tribal', 'low', 'condition', 'concern', 'fleeing', 'share', 'whole', 'paid', 'environment', 'rest', 'professor', 'little', 'hold', 'claim', 'gas', 'land', 'gillespie', 'payment', 'rain', 'proposals', 'global', 'research', 'affected', 'french', 'almost', 'damaged', 'injuries', 'construction', 'signed', 'rival', 'ice', 'deadly', 'insurance', 'efforts', 'needed', 'provide', 'companies', 'led', 'greater', 'grant', 'sign', 'mean', 'problem', 'election', 'verdict', 'ruled', 'private', 'hayden', 'treatment', 'ramallah', 'cars', 'coalition', 'dozens', 'everything', 'continued', 'representation', 'forecast', 'central', 'recorded', 'moved', 'coming', 'overall', 'sides', 'twice', 'tony', 'fired', 'steve', 'severe', 'fellow', 'rumsfeld', 'technology', 'secret', 'small', 'tape', 'statistics', 'civil', 'absolutely', 'vaughan', 'nation', 'doubles', 'resolution', 'bonn', 'cities', 'hope', 'battle', 'continues', 'review', 'direct', 'accept', 'interlaken', 'carry', 'starting', 'disaster', 'shuttle', 'pilot', 'simon', 'gun', 'winner', 'stopped', 'mountain', 'confident', 'anyone', 'richard', 'receiving', 'hotel', 'assisting', 'costello', 'ministry', 'michael', 'apparently', 'civilians', 'august', 'heading', 'conducted', 'charges', 'revealed', 'heritage', 'refugees', 'issued', 'data', 'crackdown', 'shaun', 'confirm', 'levels', 'remaining', 'yachts', 'helped', 'qc', 'territories', 'park', 'table', 'served', 'property', 'include', 'mid', 'search', 'europe', 'saw', 'winning', 'debate', 'resolve', 'markets', 'virgin', 'rescue', 'mayor', 'started', 'enter', 'knew', 'friedli', 'suffered', 'fall', 'stand', 'nice', 'determined', 'keep', 'itself', 'krishna', 'ray', 'expects', 'roads', 'body', 'season', 'negotiating', 'reduce', 'related', 'avoid', 'manslaughter', 'ball', 'vice', 'initial', 'track', 'red', 'hopes', 'above', 'leg', 'ponting', 'volunteers', 'heart', 'responsible', 'press', 'club', 'lung', 'nothing', 'remains', 'japan', 'america', 'approach', 'lower', 'fell', 'treated', 'threatened', 'guard', 'provisional', 'charge', 'cease', 'finance', 'pollock', 'tough', 'solution', 'jason', 'didn', 'victims', 'affairs', 'giuliani', 'pulled', 'operating', 'lines', 'accompanied', 'october', 'warning', 'attacked', 'strategic', 'individuals', 'spread', 'built', 'lord', 'questions', 'outlook', 'asic', 'andy', 'range', 'tuesday', 'playing', 'edge', 'suspended', 'alongside', 'wake', 'peacekeepers', 'reach', 'coach', 'showed', 'seles', 'elections', 'incident', 'seriously', 'mckenzie', 'begin', 'families', 'operation', 'victorian', 'institute', 'january', 'unemployment', 'structure', 'hearing', 'why', 'resume', 'liquidation', 'self', 'disappointed', 'successful', 'ian', 'visit', 'factory', 'delhi', 'voted', 'bit', 'wind', 'wanted', 'traditional', 'officer', 'completed', 'seeking', 'created', 'respond', 'non', 'met', 'spokeswoman', 'ceremony', 'food', 'illawarra', 'manager', 'things', 'ricky', 'networks', 'solomon', 'assault', 'finding', 'germany', 'light', 'invasion', 'single', 'summit', 'clearly', 'murder', 'wall', 'abloy', 'deployed', 'advance', 'premier', 'batsmen', 'reveal', 'investment', 'income', 'reduced', 'nearby', 'programs', 'eve', 'proteas', 'system', 'halt', 'haifa', 'oil', 'outcome', 'true', 'king', 'unrest', 'detain', 'attacking', 'clean', 'hunt', 'classic', 'whiting', 'wayne', 'amin', 'fleet', 'possibility', 'appears', 'scheduled', 'band', 'diplomatic', 'greatest', 'peres', 'live', 'billion', 'backed', 'suburbs', 'leadership', 'unity', 'philip', 'holiday', 'declaration', 'budget', 'options', 'settlement', 'products', 'extensive', 'tension', 'collapsed', 'university', 'minute', 'afroz', 'names', 'jerusalem', 'drug', 'apra', 'kashmir', 'shopping', 'real', 'handed', 'knowledge', 'yallourn', 'resign', 'employment', 'coup', 'ocean', 'often', 'nablus', 'tensions', 'students', 'gone', 'mohammad', 'austar', 'read', 'aboard', 'japanese', 'protection', 'regional', 'customers', 'follows', 'administrators', 'manufacturing', 'cave', 'recovery', 'giant', 'co', 'roof', 'happens', 'lording', 'investigating', 'gorge', 'planes', 'woman', 'felt',
                 'unit', 'targeted', 'internet', 'leave', 'gang', 'doubt', 'personnel', 'mandate', 'increased', 'acting', 'ask', 'transport', 'marine', 'battling', 'blaze', 'promised', 'actions', 'champion', 'create', 'cause', 'attempting', 'scored', 'save', 'positive', 'career', 'senate', 'numbers', 'shows', 'neil', 'grand', 'adequate', 'findings', 'swept', 'beatle', 'elders', 'criminal', 'saudi', 'honours', 'squad', 'explanation', 'secure', 'growing', 'ethnic', 'cfmeu', 'extremists', 'largest', 'pre', 'prior', 'spencer', 'singles', 'nuclear', 'raid', 'blame', 'described', 'resistance', 'ford', 'crossed', 'representing', 'natural', 'petrol', 'fatah', 'dropped', 'toll', 'corporation', 'custody', 'factions', 'injury', 'farmers', 'sarah', 'assa', 'projects', 'trial', 'ready', 'tailenders', 'jets', 'st', 'recent', 'suspect', 'races', 'speech', 'butterfly', 'boys', 'awards', 'fair', 'crews', 'scene', 'society', 'inappropriate', 'walk', 'streets', 'tree', 'prisoners', 'canberra', 'boats', 'present', 'hand', 'domestic', 'exchange', 'sheikh', 'concerns', 'switzerland', 'agreed', 'education', 'fierce', 'doug', 'traveland', 'meetings', 'presence', 'metre', 'violent', 'gambier', 'farm', 'delay', 'gary', 'sultan', 'stay', 'retired', 'colin', 'vehicle', 'westpac', 'positions', 'banking',
                 'visa', 'begun', 'masood', 'bob', 'mass', 'chosen', 'approval', 'actor', 'comment', 'necessary', 'blasted', 'sharing', 'injuring', 'fund', 'paying', 'antarctic', 'blazes', 'approached', 'returning', 'infected', 'doctor', 'threatening', 'passed', 'document', 'wednesday', 'stability', 'whatever', 'average', 'convicted', 'allegedly', 'skipper', 'proposal', 'sending', 'davis', 'focus', 'normal', 'consumers', 'aged', 'games', 'words', 'cuts', 'decisions', 'faces', 'mohammed', 'hundred', 'staying', 'project', 'publicly', 'named', 'coroner', 'target', 'relationship', 'investigate', 'title', 'improved', 'mining', 'shoes', 'rabbani', 'throughout', 'walked', 'hopman', 'cameron', 'allowed', 'channel', 'adam', 'hare', 'tie', 'previous', 'contained', 'unidentified', 'impact', 'soft', 'holding', 'owen', 'leaving', 'thing', 'putting', 'cross', 'signs', 'temporary', 'assembly', 'klusener', 'travelled', 'delivered', 'results', 'discussions', 'worked', 'became', 'heights', 'choosing', 'smaller', 'neville', 'phillips', 'ahmed', 'understanding', 'treasurer', 'harris', 'kingham', 'ability', 'provided', 'temperatures', 'telephone', 'examination', 'landed', 'voice', 'hijacked', 'mind', 'free', 'predicted', 'benares', 'male', 'paris', 'sergeant', 'archbishop', 'ban', 'locked', 'dollars', 'suggested', 'requested', 'flood', 'procedures', 'tell', 'church', 'various', 'request', 'medical', 'strachan', 'launch', 'course', 'lack', 'interview', 'occupation', 'waiting', 'fear', 'picked', 'term', 'celebrations', 'communities', 'bargaining', 'strongly', 'langer', 'happy', 'improve', 'documents', 'detail', 'credit', 'pace', 'hot', 'separate', 'headed', 'determine', 'goshen', 'guess', 'doctors', 'unfortunately', 'question', 'bringing', 'tonight', 'breaking', 'trapped', 'matthew', 'crashed', 'survived', 'clashes', 'boucher', 'hoping', 'room', 'doing', 'decades', 'seemed', 'redundancy', 'containment', 'mt', 'twenty', 'jalalabad', 'gerber', 'player', 'launceston', 'escaped', 'hamid', 'balls', 'toowoomba', 'whereabouts', 'gunships', 'aware', 'terrorists', 'firm', 'committee', 'interests', 'wing', 'indonesian', 'experts', 'finally', 'turn', 'embassy', 'headquarters', 'eventually', 'crime', 'hunter', 'ashes', 'spinner', 'humanity', 'facilities', 'path', 'effective', 'searching', 'handling', 'unable', 'anz', 'understood', 'ill', 'trading', 'sometimes', 'offered', 'effort', 'success', 'counts', 'hiv', 'follow', 'completely', 'required', 'responding', 'marines', 'henderson', 'cooperation', 'escalating', 'eliminated', 'creditors', 'history', 'abu', 'republic', 'underway', 'kissinger', 'centrelink', 'passengers', 'welcomed', 'ways', 'slightly', 'losing', 'adding', 'replied', 'francs', 'fine', 'observers', 'hopefully', 'doesn', 'hoped', 'reject', 'kieren', 'draft', 'kilometre', 'aboriginal', 'contested', 'prepare', 'stuart', 'connection', 'appropriate', 'dominance', 'identified', 'established', 'stood', 'defeat', 'prices', 'elizabeth', 'mcg', 'tactics', 'multinational', 'aip', 'badly', 'retaliatory', 'admitted', 'lose', 'accounts', 'dangerous', 'seems', 'guarantee', 'february', 'gul', 'trick', 'vehicles', 'wide', 'forestry', 'direction', 'saxet', 'nor', 'brain', 'useful', 'assets', 'highly', 'else', 'seek', 'derrick', 'video', 'scale', 'overly', 'marathon', 'site', 'interviewing', 'islamabad', 'port', 'arrests', 'lowest', 'shooting', 'hume', 'officially', 'removed', 'dealt', 'ganges', 'hardline', 'suspension', 'boost', 'defending', 'simply', 'fined', 'incidents', 'ties', 'burden', 'fled', 'appin', 'homeless', 'fully', 'becoming', 'taxpayers', 'joint', 'centuries', 'minutes', 'rudd', 'passport', 'unlikely', 'stepped', 'cells', 'hawthorne', 'quick', 'victim', 'hitting', 'generous', 'province', 'freestyle', 'veteran', 'collins', 'recovered', 'holy', 'offenders', 'applied', 'individual', 'broken', 'contract', 'loss', 'pashtun', 'closure', 'skies', 'peaceful', 'dependent', 'opportunity', 'kosovo', 'procedure', 'workforce', 'practices', 'prison', 'cow', 'mcmenamin', 'closer', 'storey', 'truss', 'rink', 'farina', 'balance', 'deployment', 'benefit', 'oversee', 'towns', 'blow', 'mother', 'nominated', 'shape', 'boje', 'hurt', 'welsh', 'losses', 'ticket', 'jakarta', 'develop', 'goodin', 'expecting', 'hornsby', 'finish', 'raises', 'london', 'native', 'fourth', 'batsman', 'trend', 'chechen', 'surrounding', 'justin', 'counsel', 'effect', 'abdul', 'fewer', 'tournament', 'majority', 'cigarettes', 'goal', 'ordered', 'knop', 'cahill', 'dixon', 'liverpool', 'discussed', 'owned', 'loyalists', 'either', 'paul', 'attempts', 'unprecedented', 'hopeful', 'recommendations', 'encouraging', 'darwin', 'highlands', 'independence', 'fuel', 'conduct', 'almao', 'kind', 'airspace', 'professional', 'properties', 'hass', 'razzano', 'driven', 'quoted', 'lance', 'particular', 'extra', 'saxeten', 'silly', 'analysis', 'century', 'claude', 'abc', 'kill', 'born', 'hanging', 'check', 'agencies', 'send', 'restore', 'film', 'wolfowitz', 'relief', 'session', 'commandos', 'code', 'alarming', 'original', 'stock', 'assessment', 'peel', 'preparation', 'appear', 'penalty', 'weak', 'agha', 'verdicts', 'operate', 'relatively', 'row', 'tax', 'ahmad', 'trained', 'happen', 'gambill', 'prove', 'swans', 'erupted', 'consider', 'kashmiri', 'dismissed', 'allies', 'queen', 'scores', 'instead', 'pair', 'competition', 'ward', 'couldn', 'spending', 'pretty', 'willingness', 'specific', 'center', 'german', 'unclear', 'activities', 'kirsten', 'partner', 'levelled', 'buried', 'inspector', 'heads', 'older', 'bat', 'murray', 'refugee', 'traffic', 'reason', 'forget', 'welfare', 'represents', 'fresh', 'shadow', 'aids', 'gutnick', 'gilchrist', 'calm', 'historic', 'negotiate', 'written', 'double', 'sanctions', 'lleyton', 'gain', 'fireworks', 'release', 'midwives', 'spent', 'facing', 'investigations', 'soccer', 'valley', 'nail', 'cnn', 'dozen', 'dangers', 'bacteria', 'sexual', 'apache', 'ease', 'establish', 'cash', 'ambulance', 'estimate', 'included', 'tests', 'supported', 'rushed', 'account', 'payne', 'imf', 'handicap', 'user', 'substantial', 'beatles', 'sullivan', 'fifth', 'possibly', 'toward', 'answer', 'prudential', 'couple', 'mastermind', 'implications', 'moment', 'hasn', 'involvement', 'redmond', 'view', 'selectors', 'involving', 'bond', 'let', 'amwu', 'missiles', 'wages', 'ship', 'european', 'handled', 'recently', 'accord', 'camps', 'mistakes', 'employee', 'grenades', 'thousand', 'route', 'rubber', 'alternative', 'showing', 'camp', 'internal', 'tight', 'brazil', 'tourism', 'enterprise', 'hiding', 'subject', 'angry', 'critical', 'videotape', 'huegill', 'balmer', 'electricity', 'ministerial', 'acknowledged', 'palace', 'reaction', 'planned', 'broke', 'suspects', 'dramatic', 'jan', 'causing', 'intimidation', 'crashing', 'moussaoui', 'discuss', 'trouble', 'reportedly', 'failing', 'ending', 'standing', 'arthurs', 'macfarlane', 'finalised', 'care', 'progress', 'lennon', 'alan', 'asking', 'particularly', 'rising', 'timed', 'email', 'paper', 'sharp', 'casinos', 'safe', 'federation', 'fundamental', 'association', 'employed', 'spills', 'magistrate', 'mohamad', 'diagnosed', 'eligible', 'starts', 'supply', 'intervention', 'escude', 'trends', 'graham', 'carl', 'symbols', 'author', 'confessed', 'zaman', 'adopted', 'ran', 'genetically', 'nato', 'clash', 'saa', 'begins', 'turning', 'village', 'stefan', 'perpetrators', 'mainly', 'scouring', 'jirga', 'proves', 'shell', 'crowded', 'ranging', 'hear', 'trounson', 'robertson', 'advani', 'remember', 'upsurge',
                 'stumps', 'applications', 'urged', 'tougher', 'suggests', 'fans', 'formal', 'cope', 'templeton', 'counting', 'gabriel', 'slowing', 'powered', 'partnership', 'payments', 'travellers', 'villawood', 'rare', 'welcome', 'spin', 'destruction', 'thunderstorm', 'nautical', 'speculation', 'requests', 'faith', 'africans', 'pleased', 'ourselves', 'raise', 'musical', 'conspirators', 'incentive', 'upper', 'solvency', 'rioting', 'values', 'tommy', 'miles', 'withdrawal', 'initially', 'palmerston', 'downgrade', 'stance', 'arrivals', 'shifted', 'amazon', 'catch', 'equipment', 'tv', 'aviation', 'brussels', 'leak', 'kennedy', 'dick', 'chase', 'imagine', 'daily', 'wouldn', 'vowed', 'slips', 'hayward', 'sport', 'asian', 'hawke', 'gibbons', 'stoltenberg', 'guy', 'square', 'sinai', 'events', 'evening', 'dfat', 'olivier', 'monitored', 'franklin', 'model', 'feared', 'tailender', 'knife', 'tunnels', 'currency', 'likewise', 'britt', 'maintains', 'presidential', 'plants', 'importance', 'escalate', 'peru', 'housing', 'pervez', 'cairns', 'surprise', 'criticised', 'passenger', 'rfds', 'standards', 'solo', 'locations', 'nicorette', 'personally', 'towers', 'downturn', 'informed', 'dealing', 'langdale', 'wife', 'fought', 'arresting', 'regulation', 'luck', 'defunct', 'skippered', 'arab', 'assured', 'ferguson', 'reuters', 'wearing', 'perfect', 'cases', 'discussing', 'democracy', 'sheet', 'salaries', 'guards', 'undertaken', 'transitional', 'generally', 'tyco', 'abegglen', 'pitched', 'threw', 'maguire', 'score', 'institutions', 'emissions', 'broadcast', 'utn', 'wild', 'transferred', 'habeel', 'marked', 'eyes', 'protected', 'default', 'stephan', 'players', 'duck', 'satellite', 'considering', 'religious', 'lieutenant', 'mitsubishi', 'cover', 'declaring', 'hospitals', 'soil', 'belief', 'placing', 'device', 'shimon', 'grass', 'interested', 'chasing', 'stronger', 'announcement', 'bbc', 'perhaps', 'loya', 'rehman', 'alei', 'arriving', 'shaky', 'dewar', 'terrible', 'protesters', 'arabs', 'rescued', 'easy', 'everyone', 'reporters', 'owner', 'arbitration', 'iraq', 'terminal', 'stepping', 'visas', 'struggling', 'defendants', 'friend', 'advertising', 'decade', 'vessel', 'haitian', 'astronauts', 'shown', 'counter', 'battleground', 'girl', 'coastal', 'blast', 'development', 'evacuated', 'danger', 'contact', 'forecasting', 'saadi', 'infantry', 'consequences', 'unpredictable', 'lucky', 'offering', 'riots', 'recognised', 'memory', 'militias', 'silent', 'prohibited', 'operator', 'climate', 'comeback', 'communications', 'nasty', 'projections', 'combination', 'crack', 'ongoing', 'prix', 'export', 'sought', 'siege', 'attendants', 'nelson', 'legislation', 'jackson', 'occur', 'limits', 'alcohol', 'allowing', 'termination', 'strategy', 'unpaid', 'runway', 'casualties', 'ranked', 'date', 'firms', 'computer', 'enemy', 'timing', 'russell', 'demanded', 'islam', 'china', 'ashcroft', 'resigned', 'parliamentary', 'infrastructure', 'suburb', 'whilst', 'helen', 'bail', 'arm', 'funding', 'yunis', 'holds', 'relation', 'harm', 'waging', 'softer', 'bombed', 'proceed', 'earning', 'apply', 'deciding', 'living', 'corowa', 'prompted', 'canada', 'published', 'chemical', 'aziz', 'dollar', 'outsourcing', 'dialogue', 'extremely', 'ron', 'stray', 'beattie', 'gets', 'summer', 'behalf', 'computers', 'razzak', 'embryo', 'walker', 'policemen', 'globe', 'survival', 'ivf', 'type', 'approved', 'plant', 'split', 'restructuring', 'catches', 'recommendation', 'factors', 'interrogation', 'escalated', 'baker', 'wran', 'favour',
                 'gunman', 'returns', 'ideas', 'enormous', 'defeated', 'symptoms', 'blocks', 'compound', 'provincial', 'delayed', 'paktika', 'risen', 'points', 'fatality', 'obviously', 'champions', 'mounted', 'mosque', 'committed', 'music', 'surprised', 'boston', 'pr', 'negotiator', 'bradford', 'shop', 'profits', 'administrator', 'crimes', 'hawkesbury', 'vajpayee', 'personality', 'surrounded', 'resort', 'flee', 'swedish', 'acdt', 'join', 'fears', 'detained', 'sets', 'paceman', 'mount', 'todd', 'seem', 'arrival', 'myself', 'lali',
                 'judge', 'spain', 'defended', 'sweeping', 'hat', 'overrun', 'uncertain', 'junior', 'mountainous', 'previously', 'occurred', 'teams', 'example', 'pennsylvania', 'breathing', 'perkins', 'stores', 'totally', 'parents', 'teenager', 'ensuring', 'oval', 'happening', 'marks', 'vigil', 'aging', 'haven', 'june', 'fort', 'witness', 'bowled', 'christian', 'effectively', 'trials', 'england', 'option', 'sponsored', 'dying', 'prize', 'deadline', 'competitive', 'considered', 'controlled', 'exactly', 'maxi', 'honour', 'megawati', 'teenage', 'task', 'razor', 'consistent', 'resolved', 'canyon', 'ernst', 'bowlers', 'push', 'deserve', 'quit', 'contributions', 'eighth', 'heavily', 'appointment', 'protest', 'journey', 'card', 'lone', 'repeated', 'brother', 'ones', 'firemen', 'engines', 'interviewed', 'lew', 'uss', 'authorising', 'slip', 'command', 'causes', 'decline', 'collect', 'accountancy', 'acquitted', 'escape', 'principle', 'evil', 'priest', 'serve', 'linked', 'black', 'complacency', 'guide', 'funds', 'claiming', 'ntini', 'swimming', 'yassin', 'monetary', 'guarding', 'widespread', 'scarfe', 'aspects', 'presidency', 'miami', 'isolated', 'transfer', 'reasonably', 'impossible', 'fate', 'lindsay', 'obese', 'dickie', 'consent', 'augusta', 'poor', 'indicated', 'blamed', 'ruling', 'aimed', 'cheney', 'obesity', 'limited', 'winter', 'bayliss', 'learn', 'identify', 'decisive', 'congress', 'frank', 'midnight', 'hills', 'includes', 'wiget', 'neighbouring', 'setting', 'settler', 'democratic', 'reminded', 'johnston', 'article', 'talented', 'adult', 'once', 'taylor', 'reasons', 'sheldon', 'aim', 'peacekeeping', 'social', 'attended', 'stands', 'minor', 'masterminding', 'providing', 'remove', 'comply', 'meant', 'dominant', 'anybody', 'denies', 'raduyev', 'aiming', 'attempted', 'irrelevant', 'hectares', 'dawn', 'assist', 'appealed', 'profit', 'remote', 'commercial', 'love', 'indiana', 'sitting', 'coincide', 'swimmer', 'admits', 'adults', 'mounting', 'forests', 'blew', 'reputation', 'goes', 'shoalhaven', 'seeing', 'thick', 'deliberate', 'khan', 'striking', 'tribute', 'earth', 'principles', 'finishing', 'reading', 'christians', 'faced', 'embryos', 'strait', 'ali', 'panel', 'lunchtime', 'tower', 'defuse', 'securities', 'motor', 'paedophiles', 'managers', 'spill', 'worm', 'settlers', 'fly']
        self.assertTrue(a, True)
        self.assertEqual(wv.index_to_key, ans_key)

    def test_writeWordEmbedding(self):
        wv = readWordEmbedding('my_model1.bin')
        writeWordEmbedding(wv,'temp_model.bin')
        wv1=readWordEmbedding('temp_model.bin')
        a, b = False, False
        if isinstance(wv1, Vectors):
            a = True
        ans_key = ['the', 'to', 'of', 'in', 'and', 'he', 'is', 'for', 'on', 'said', 'that', 'has', 'says', 'was',
                   'have', 'it', 'be', 'are', 'with',
                   'will', 'at', 'mr', 'from', 'by', 'we', 'been', 'as', 'an', 'not', 'his', 'but', 'they', 'after',
                   'were', 'had', 'there', 'new',
                   'this', 'australian', 'australia', 'who', 'people', 'palestinian', 'their', 'government', 'two',
                   'up', 'south', 'us', 'which',
                   'year', 'one', 'about', 'out', 'if', 'also', 'more', 'when', 'its', 'into', 'would', 'first',
                   'against', 'last', 'israeli', 'minister',
                   'arafat', 'all', 'over', 'afghanistan', 'three', 'united', 'no', 'world', 'police', 'or', 'than',
                   'attacks', 'before', 'fire', 'day', 'security', 'some', 'states', 'you', 'could', 'them', 'say',
                   'today', 'now', 'told', 'time', 'any', 'very', 'laden', 'bin', 'just', 'can', 'what', 'president',
                   'sydney', 'company', 'still', 'four', 'man', 'killed', 'taliban', 'al', 'forces', 'our', 'around',
                   'west', 'being', 'days', 'old', 'other', 'officials', 'where', 'so', 'test', 'qaeda', 'israel',
                   'next', 'per', 'general', 'think', 'federal', 'she', 'force', 'cent', 'workers', 'leader',
                   'yesterday', 'under', 'hamas', 'take', 'state', 'him', 'those', 'bank', 'years', 'back', 'meeting',
                   'suicide', 'made', 'morning', 'action', 'down', 'commission', 're', 'pakistan', 'international',
                   'attack', 'centre', 'afghan', 'group', 'city', 'well', 'through', 'military', 'members', 'while',
                   'number', 'five', 'called', 'local', 'area', 'qantas', 'gaza', 'week', 'union', 'national', 'since',
                   'hours', 'wales', 'september', 'including', 'another', 'east', 'report', 'night', 'off', 'north',
                   'should', 'six', 'staff', 'between', 'these', 'get', 'second', 'earlier', 'war', 'go', 'islamic',
                   'further', 'defence', 'end', 'months', 'do', 'because', 'authority', 'foreign', 'going', 'power',
                   'work', 'areas', 'near', 'team', 'sharon', 'during', 'died', 'month', 'only', 'many', 'india', 'way',
                   'eight', 'know', 'metres', 'match', 'good', 'make', 've', 'melbourne', 'northern', 'spokesman',
                   'claims', 'former', 'left', 'air', 'most', 'support', 'osama', 'peace', 'like', 'authorities',
                   'prime', 'given', 'am', 'ago', 'saying', 'set', 'expected', 'tora', 'put', 'bora', 'looking', 'come',
                   'place', 'militants', 'fighters', 'several', 'children', 'meanwhile', 'injured', 'christmas',
                   'groups', 'indian', 'found', 'unions', 'arrested', 'royal', 'river', 'troops', 'child', 'africa',
                   'talks', 'official', 'whether', 'interim', 'reports', 'then', 'hospital', 'terrorist', 'yasser',
                   'part', 'industrial', 'don', 'start', 'how', 'statement', 'leaders', 'third', 'early', 'senior',
                   'terrorism', 'economy', 'mountains', 'weather', 'hit', 'million', 'believe', 'trying', 'family',
                   'both', 'john', 'army', 'pay', 'court', 'radio', 'public', 'dr', 'asked', 'control', 'lead',
                   'pressure', 'best', 'long', 'adelaide', 'chief', 'following', 'however', 'agreement', 'help', 'few',
                   'house', 'play', 'labor', 'australians', 'arrest', 'better', 'want', 'does', 'firefighters', 'high',
                   'need', 'close', 'service', 'community', 'taken', 'confirmed', 'queensland', 'services', 'overnight',
                   'process', 'opposition', 'williams', 'must', 'secretary', 'information', 'believed', 'came',
                   'detainees', 'won', 'governor', 'held', 'shot', 'damage', 'possible', 'her', 'hicks', 'nations',
                   'much', 'pentagon', 'peter', 'party', 'did', 'released', 'win', 'maintenance', 'took', 'brought',
                   'british', 'accused', 'safety', 'armed', 'kandahar', 'winds', 'despite', 'even', 'such', 'across',
                   'eastern', 'violence', 'weekend', 'return', 'conditions', 'without', 'building', 'lot', 'asylum',
                   'dead', 'cut', 'york', 'change', 'hill', 'director', 'council', 'airline', 'got', 'far', 'news',
                   'lee', 'waugh', 'trade', 'southern',
                   'crew', 'continuing', 'monday', 'captured', 'fires', 'see', 'race', 'economic', 'strong', 'call',
                   'anti', 'emergency', 'cricket', 'region', 'aircraft', 'palestinians', 'men', 'home', 'training',
                   'seekers', 'working', 'strip', 'country', 'american', 'david', 'board', 'role', 'here', 'december',
                   'too', 'fighting', 'plans', 'industry', 'george', 'charged', 'act', 'health', 'bush', 'received',
                   'key', 'alliance', 'rate', 'past', 'administration', 'bureau', 'used', 'head', 'offer', 'water',
                   'zinni', 'town', 'within', 'boat', 'decision', 'zealand', 'least', 'israelis', 'strikes', 'britain',
                   'line', 'department', 'soldiers', 'hih', 'station', 'issue', 'downer', 'leading', 'use', 'major',
                   'person', 'operations', 'hundreds', 'stop', 'final', 'parliament', 'known', 'captain', 'legal',
                   'large', 'airport', 'risk', 'your', 'may', 'later', 'give', 'ahead', 'officers', 'tomorrow', 'half',
                   'due', 'un', 'series', 'laws', 'able', 'interest', 'every', 'homes', 'taking', 'weapons', 'coast',
                   'behind', 'hollingworth', 'policy', 'network', 'western', 'kabul', 'pm', 'great', 'latest', 'late',
                   'plane', 'my', 'remain', 'might', 'right', 'shane', 'death', 'position', 'already', 'heard',
                   'deaths', 'forced', 'life', 'hard', 'failed', 'seen', 'continue', 'towards', 'along', 'jihad',
                   'side', 'timor', 'abuse', 'territory', 'special', 'others', 'guilty', 'campaign', 'bill', 'storm',
                   'same', 'flight', 'concerned', 'victory', 'cup', 'jobs', 'thought', 'event', 'me', 'point', 'really',
                   'washington', 'member', 'buildings', 'november', 'case', 'sunday', 'weeks', 'bombing', 'mcgrath',
                   'bowler', 'matter', 'th', 'innings', 'helicopters', 'bus', 'envoy', 'details', 'countries', 'likely',
                   'middle', 'canyoning', 'move', 'rates', 'situation', 'cabinet', 'again', 'capital', 'woomera',
                   'seven', 'bichel', 'warne', 'mark', 'launched', 'according', 'detention', 'enough', 'bombings',
                   'important', 'space', 'hour', 'office', 'wants', 'boy', 'human', 'adventure', 'perth', 'women',
                   'young', 'political', 'deal', 'asio', 'based', 'claimed', 'commonwealth', 'evidence', 'sure',
                   'justice', 'swiss', 'jail', 'reported', 'aedt', 'caught', 'own', 'movement', 'mission', 'african',
                   'disease', 'added', 'show', 'raids', 'blue', 'opened', 'run', 'money', 'forward', 'immediately',
                   'guides', 'rule', 'top', 'carried', 'freeze', 'warplanes', 'targets', 'cancer', 'ms', 'dispute',
                   'wicket', 'times', 'face', 'march', 'always', 'investigation', 'border', 'flying', 'full',
                   'allegations', 'sector', 'financial', 'although', 'growth', 'ground', 'burning', 'result', 'order',
                   'crash', 'planning', 'break', 'island', 'job', 'become', 'carrying', 'find', 'using', 'access',
                   'beat', 'prepared', 'each', 'certainly', 'banks', 'reached', 'proposed', 'probably', 'collapse',
                   'relations', 'inside', 'reserve', 'allow', 'tourists', 'militia', 'organisation', 'radical',
                   'militant', 'afp', 'road', 'wave', 'different', 'executive', 'energy', 'clear', 'serious',
                   'responsibility', 'hewitt', 'post', 'making', 'inquiry', 'sent', 'harrison', 'suspected',
                   'surrender', 'trees', 'management', 'thousands', 'thursday', 'needs', 'bowling', 'future',
                   'rejected', 'outside', 'opening', 'travel', 'kilometres', 'short', 'killing', 'ariel', 'quickly',
                   'howard', 'went', 'drop', 'bid', 'sex', 'himself', 'declared', 'comes', 'fight', 'lives', 'soon',
                   'ansett', 'immigration', 'caves', 'tried', 'stage', 'argentina', 'believes', 'senator', 'flights',
                   'calls', 'program', 'getting', 'until', 'alleged', 'sentence', 'circumstances', 'television',
                   'quite', 'annual', 'rural', 'wounded', 'try', 'open', 'attempt', 'white', 'terms', 'ruddock',
                   'among', 'offices', 'sea', 'currently', 'tanks', 'available', 'sources', 'ended', 'donald',
                   'anything', 'refused', 'look', 'figures', 'changes', 'means', 'measures', 'alexander', 'shortly',
                   'yacht', 'issues', 'form', 'july', 'victoria', 'running', 'bombers', 'agency', 'address', 'response',
                   'gunmen', 'karzai', 'meet', 'happened', 'directors', 'actually', 'jewish', 'yet', 'something',
                   'done', 'wickets', 'residents', 'returned', 'destroyed', 'brisbane', 'announced', 'accident',
                   'warned', 'understand', 'parts', 'rise', 'decided', 'ministers', 'witnesses', 'biggest', 'parties',
                   'nine', 'list', 'source', 'kallis', 'fighter', 'school', 'increase', 'rights', 'caused', 'allan',
                   'tour', 'highway', 'deputy',
                   'media', 'commissioner', 'arrived', 'facility', 'away', 'elected', 'though', 'advice', 'supporters',
                   'ses', 'total', 'pacific', 'cost', 'law', 'wage', 'fact', 'difficult', 'prevent', 'began',
                   'confidence', 'negotiations', 'blake', 'huge', 'rafter', 'beyond', 'rather', 'beginning',
                   'sir', 'glenn', 'business', 'afternoon', 'struck', 'intelligence', 'ensure', 'virus', 'period',
                   'big', 'tennis', 'islands', 'car', 'having', 'commander', 'appeared', 'step', 'powell', 'strike',
                   'crowd', 'whose', 'expect', 'ever', 'runs', 'commanders', 'suharto', 'airlines', 'market', 'nauru',
                   'll', 'less', 'attorney', 'expressed', 'gave', 'worst', 'involved', 'gives', 'followed', 'recession',
                   'musharraf', 'robert', 'field', 'friday', 'bomb', 'hobart', 'tragedy', 'terror', 'smoke',
                   'potential', 'plan', 'played', 'feel', 'reid', 'places', 'speaking', 'costs', 'anthony', 'entered',
                   'muslim', 'hearings', 'lockett', 'helicopter', 'front', 'storms', 'organisations', 'crisis', 'jenin',
                   'ambush', 'quarter', 'level', 'conflict', 'base', 'zimbabwe', 'endeavour', 'chance', 'macgill',
                   'longer', 'giving', 'employees', 'nearly', 'explosives', 'assistance', 'yes', 'geoff', 'changed',
                   'chairman', 'vote', 'criticism', 'heavy', 'main', 'complex', 'threat', 'game', 'brett', 'calling',
                   'jacques', 'lost', 'population', 'bring', 'study', 'saturday', 'russian', 'fast', 'round',
                   'indonesia', 'protect', 'sort', 'daryl', 'current', 'never', 'together', 'decide', 'survey',
                   'conference', 'massive', 'powers', 'boxing', 'anglican', 'labour', 'crean', 'americans',
                   'themselves', 'martin', 'finished', 'placed', 'denied', 'son', 'entitlements', 'ballot', 'cannot',
                   'significant', 'pakistani', 'debt', 'france', 'tasmania', 'trip', 'receive', 'flames', 'record',
                   'activity', 'bomber', 'closed', 'problems', 'tribal', 'low', 'condition', 'concern', 'fleeing',
                   'share', 'whole', 'paid', 'environment', 'rest', 'professor', 'little', 'hold', 'claim', 'gas',
                   'land', 'gillespie', 'payment', 'rain', 'proposals', 'global', 'research', 'affected', 'french',
                   'almost', 'damaged', 'injuries', 'construction', 'signed', 'rival', 'ice', 'deadly', 'insurance',
                   'efforts', 'needed', 'provide', 'companies', 'led', 'greater', 'grant', 'sign', 'mean', 'problem',
                   'election', 'verdict', 'ruled', 'private', 'hayden', 'treatment', 'ramallah', 'cars', 'coalition',
                   'dozens', 'everything', 'continued', 'representation', 'forecast', 'central', 'recorded', 'moved',
                   'coming', 'overall', 'sides', 'twice', 'tony', 'fired', 'steve', 'severe', 'fellow', 'rumsfeld',
                   'technology', 'secret', 'small', 'tape', 'statistics', 'civil', 'absolutely', 'vaughan', 'nation',
                   'doubles', 'resolution', 'bonn', 'cities', 'hope', 'battle', 'continues', 'review', 'direct',
                   'accept', 'interlaken', 'carry', 'starting', 'disaster', 'shuttle', 'pilot', 'simon', 'gun',
                   'winner', 'stopped', 'mountain', 'confident', 'anyone', 'richard', 'receiving', 'hotel', 'assisting',
                   'costello', 'ministry', 'michael', 'apparently', 'civilians', 'august', 'heading', 'conducted',
                   'charges', 'revealed', 'heritage', 'refugees', 'issued', 'data', 'crackdown', 'shaun', 'confirm',
                   'levels', 'remaining', 'yachts', 'helped', 'qc', 'territories', 'park', 'table', 'served',
                   'property', 'include', 'mid', 'search', 'europe', 'saw', 'winning', 'debate', 'resolve', 'markets',
                   'virgin', 'rescue', 'mayor', 'started', 'enter', 'knew', 'friedli', 'suffered', 'fall', 'stand',
                   'nice', 'determined', 'keep', 'itself', 'krishna', 'ray', 'expects', 'roads', 'body', 'season',
                   'negotiating', 'reduce', 'related', 'avoid', 'manslaughter', 'ball', 'vice', 'initial', 'track',
                   'red', 'hopes', 'above', 'leg', 'ponting', 'volunteers', 'heart', 'responsible', 'press', 'club',
                   'lung', 'nothing', 'remains', 'japan', 'america', 'approach', 'lower', 'fell', 'treated',
                   'threatened', 'guard', 'provisional', 'charge', 'cease', 'finance', 'pollock', 'tough', 'solution',
                   'jason', 'didn', 'victims', 'affairs', 'giuliani', 'pulled', 'operating', 'lines', 'accompanied',
                   'october', 'warning', 'attacked', 'strategic', 'individuals', 'spread', 'built', 'lord', 'questions',
                   'outlook', 'asic', 'andy', 'range', 'tuesday', 'playing', 'edge', 'suspended', 'alongside', 'wake',
                   'peacekeepers', 'reach', 'coach', 'showed', 'seles', 'elections', 'incident', 'seriously',
                   'mckenzie', 'begin', 'families', 'operation', 'victorian', 'institute', 'january', 'unemployment',
                   'structure', 'hearing', 'why', 'resume', 'liquidation', 'self', 'disappointed', 'successful', 'ian',
                   'visit', 'factory', 'delhi', 'voted', 'bit', 'wind', 'wanted', 'traditional', 'officer', 'completed',
                   'seeking', 'created', 'respond', 'non', 'met', 'spokeswoman', 'ceremony', 'food', 'illawarra',
                   'manager', 'things', 'ricky', 'networks', 'solomon', 'assault', 'finding', 'germany', 'light',
                   'invasion', 'single', 'summit', 'clearly', 'murder', 'wall', 'abloy', 'deployed', 'advance',
                   'premier', 'batsmen', 'reveal', 'investment', 'income', 'reduced', 'nearby', 'programs', 'eve',
                   'proteas', 'system', 'halt', 'haifa', 'oil', 'outcome', 'true', 'king', 'unrest', 'detain',
                   'attacking', 'clean', 'hunt', 'classic', 'whiting', 'wayne', 'amin', 'fleet', 'possibility',
                   'appears', 'scheduled', 'band', 'diplomatic', 'greatest', 'peres', 'live', 'billion', 'backed',
                   'suburbs', 'leadership', 'unity', 'philip', 'holiday', 'declaration', 'budget', 'options',
                   'settlement', 'products', 'extensive', 'tension', 'collapsed', 'university', 'minute', 'afroz',
                   'names', 'jerusalem', 'drug', 'apra', 'kashmir', 'shopping', 'real', 'handed', 'knowledge',
                   'yallourn', 'resign', 'employment', 'coup', 'ocean', 'often', 'nablus', 'tensions', 'students',
                   'gone', 'mohammad', 'austar', 'read', 'aboard', 'japanese', 'protection', 'regional', 'customers',
                   'follows', 'administrators', 'manufacturing', 'cave', 'recovery', 'giant', 'co', 'roof', 'happens',
                   'lording', 'investigating', 'gorge', 'planes', 'woman', 'felt',
                   'unit', 'targeted', 'internet', 'leave', 'gang', 'doubt', 'personnel', 'mandate', 'increased',
                   'acting', 'ask', 'transport', 'marine', 'battling', 'blaze', 'promised', 'actions', 'champion',
                   'create', 'cause', 'attempting', 'scored', 'save', 'positive', 'career', 'senate', 'numbers',
                   'shows', 'neil', 'grand', 'adequate', 'findings', 'swept', 'beatle', 'elders', 'criminal', 'saudi',
                   'honours', 'squad', 'explanation', 'secure', 'growing', 'ethnic', 'cfmeu', 'extremists', 'largest',
                   'pre', 'prior', 'spencer', 'singles', 'nuclear', 'raid', 'blame', 'described', 'resistance', 'ford',
                   'crossed', 'representing', 'natural', 'petrol', 'fatah', 'dropped', 'toll', 'corporation', 'custody',
                   'factions', 'injury', 'farmers', 'sarah', 'assa', 'projects', 'trial', 'ready', 'tailenders', 'jets',
                   'st', 'recent', 'suspect', 'races', 'speech', 'butterfly', 'boys', 'awards', 'fair', 'crews',
                   'scene', 'society', 'inappropriate', 'walk', 'streets', 'tree', 'prisoners', 'canberra', 'boats',
                   'present', 'hand', 'domestic', 'exchange', 'sheikh', 'concerns', 'switzerland', 'agreed',
                   'education', 'fierce', 'doug', 'traveland', 'meetings', 'presence', 'metre', 'violent', 'gambier',
                   'farm', 'delay', 'gary', 'sultan', 'stay', 'retired', 'colin', 'vehicle', 'westpac', 'positions',
                   'banking',
                   'visa', 'begun', 'masood', 'bob', 'mass', 'chosen', 'approval', 'actor', 'comment', 'necessary',
                   'blasted', 'sharing', 'injuring', 'fund', 'paying', 'antarctic', 'blazes', 'approached', 'returning',
                   'infected', 'doctor', 'threatening', 'passed', 'document', 'wednesday', 'stability', 'whatever',
                   'average', 'convicted', 'allegedly', 'skipper', 'proposal', 'sending', 'davis', 'focus', 'normal',
                   'consumers', 'aged', 'games', 'words', 'cuts', 'decisions', 'faces', 'mohammed', 'hundred',
                   'staying', 'project', 'publicly', 'named', 'coroner', 'target', 'relationship', 'investigate',
                   'title', 'improved', 'mining', 'shoes', 'rabbani', 'throughout', 'walked', 'hopman', 'cameron',
                   'allowed', 'channel', 'adam', 'hare', 'tie', 'previous', 'contained', 'unidentified', 'impact',
                   'soft', 'holding', 'owen', 'leaving', 'thing', 'putting', 'cross', 'signs', 'temporary', 'assembly',
                   'klusener', 'travelled', 'delivered', 'results', 'discussions', 'worked', 'became', 'heights',
                   'choosing', 'smaller', 'neville', 'phillips', 'ahmed', 'understanding', 'treasurer', 'harris',
                   'kingham', 'ability', 'provided', 'temperatures', 'telephone', 'examination', 'landed', 'voice',
                   'hijacked', 'mind', 'free', 'predicted', 'benares', 'male', 'paris', 'sergeant', 'archbishop', 'ban',
                   'locked', 'dollars', 'suggested', 'requested', 'flood', 'procedures', 'tell', 'church', 'various',
                   'request', 'medical', 'strachan', 'launch', 'course', 'lack', 'interview', 'occupation', 'waiting',
                   'fear', 'picked', 'term', 'celebrations', 'communities', 'bargaining', 'strongly', 'langer', 'happy',
                   'improve', 'documents', 'detail', 'credit', 'pace', 'hot', 'separate', 'headed', 'determine',
                   'goshen', 'guess', 'doctors', 'unfortunately', 'question', 'bringing', 'tonight', 'breaking',
                   'trapped', 'matthew', 'crashed', 'survived', 'clashes', 'boucher', 'hoping', 'room', 'doing',
                   'decades', 'seemed', 'redundancy', 'containment', 'mt', 'twenty', 'jalalabad', 'gerber', 'player',
                   'launceston', 'escaped', 'hamid', 'balls', 'toowoomba', 'whereabouts', 'gunships', 'aware',
                   'terrorists', 'firm', 'committee', 'interests', 'wing', 'indonesian', 'experts', 'finally', 'turn',
                   'embassy', 'headquarters', 'eventually', 'crime', 'hunter', 'ashes', 'spinner', 'humanity',
                   'facilities', 'path', 'effective', 'searching', 'handling', 'unable', 'anz', 'understood', 'ill',
                   'trading', 'sometimes', 'offered', 'effort', 'success', 'counts', 'hiv', 'follow', 'completely',
                   'required', 'responding', 'marines', 'henderson', 'cooperation', 'escalating', 'eliminated',
                   'creditors', 'history', 'abu', 'republic', 'underway', 'kissinger', 'centrelink', 'passengers',
                   'welcomed', 'ways', 'slightly', 'losing', 'adding', 'replied', 'francs', 'fine', 'observers',
                   'hopefully', 'doesn', 'hoped', 'reject', 'kieren', 'draft', 'kilometre', 'aboriginal', 'contested',
                   'prepare', 'stuart', 'connection', 'appropriate', 'dominance', 'identified', 'established', 'stood',
                   'defeat', 'prices', 'elizabeth', 'mcg', 'tactics', 'multinational', 'aip', 'badly', 'retaliatory',
                   'admitted', 'lose', 'accounts', 'dangerous', 'seems', 'guarantee', 'february', 'gul', 'trick',
                   'vehicles', 'wide', 'forestry', 'direction', 'saxet', 'nor', 'brain', 'useful', 'assets', 'highly',
                   'else', 'seek', 'derrick', 'video', 'scale', 'overly', 'marathon', 'site', 'interviewing',
                   'islamabad', 'port', 'arrests', 'lowest', 'shooting', 'hume', 'officially', 'removed', 'dealt',
                   'ganges', 'hardline', 'suspension', 'boost', 'defending', 'simply', 'fined', 'incidents', 'ties',
                   'burden', 'fled', 'appin', 'homeless', 'fully', 'becoming', 'taxpayers', 'joint', 'centuries',
                   'minutes', 'rudd', 'passport', 'unlikely', 'stepped', 'cells', 'hawthorne', 'quick', 'victim',
                   'hitting', 'generous', 'province', 'freestyle', 'veteran', 'collins', 'recovered', 'holy',
                   'offenders', 'applied', 'individual', 'broken', 'contract', 'loss', 'pashtun', 'closure', 'skies',
                   'peaceful', 'dependent', 'opportunity', 'kosovo', 'procedure', 'workforce', 'practices', 'prison',
                   'cow', 'mcmenamin', 'closer', 'storey', 'truss', 'rink', 'farina', 'balance', 'deployment',
                   'benefit', 'oversee', 'towns', 'blow', 'mother', 'nominated', 'shape', 'boje', 'hurt', 'welsh',
                   'losses', 'ticket', 'jakarta', 'develop', 'goodin', 'expecting', 'hornsby', 'finish', 'raises',
                   'london', 'native', 'fourth', 'batsman', 'trend', 'chechen', 'surrounding', 'justin', 'counsel',
                   'effect', 'abdul', 'fewer', 'tournament', 'majority', 'cigarettes', 'goal', 'ordered', 'knop',
                   'cahill', 'dixon', 'liverpool', 'discussed', 'owned', 'loyalists', 'either', 'paul', 'attempts',
                   'unprecedented', 'hopeful', 'recommendations', 'encouraging', 'darwin', 'highlands', 'independence',
                   'fuel', 'conduct', 'almao', 'kind', 'airspace', 'professional', 'properties', 'hass', 'razzano',
                   'driven', 'quoted', 'lance', 'particular', 'extra', 'saxeten', 'silly', 'analysis', 'century',
                   'claude', 'abc', 'kill', 'born', 'hanging', 'check', 'agencies', 'send', 'restore', 'film',
                   'wolfowitz', 'relief', 'session', 'commandos', 'code', 'alarming', 'original', 'stock', 'assessment',
                   'peel', 'preparation', 'appear', 'penalty', 'weak', 'agha', 'verdicts', 'operate', 'relatively',
                   'row', 'tax', 'ahmad', 'trained', 'happen', 'gambill', 'prove', 'swans', 'erupted', 'consider',
                   'kashmiri', 'dismissed', 'allies', 'queen', 'scores', 'instead', 'pair', 'competition', 'ward',
                   'couldn', 'spending', 'pretty', 'willingness', 'specific', 'center', 'german', 'unclear',
                   'activities', 'kirsten', 'partner', 'levelled', 'buried', 'inspector', 'heads', 'older', 'bat',
                   'murray', 'refugee', 'traffic', 'reason', 'forget', 'welfare', 'represents', 'fresh', 'shadow',
                   'aids', 'gutnick', 'gilchrist', 'calm', 'historic', 'negotiate', 'written', 'double', 'sanctions',
                   'lleyton', 'gain', 'fireworks', 'release', 'midwives', 'spent', 'facing', 'investigations', 'soccer',
                   'valley', 'nail', 'cnn', 'dozen', 'dangers', 'bacteria', 'sexual', 'apache', 'ease', 'establish',
                   'cash', 'ambulance', 'estimate', 'included', 'tests', 'supported', 'rushed', 'account', 'payne',
                   'imf', 'handicap', 'user', 'substantial', 'beatles', 'sullivan', 'fifth', 'possibly', 'toward',
                   'answer', 'prudential', 'couple', 'mastermind', 'implications', 'moment', 'hasn', 'involvement',
                   'redmond', 'view', 'selectors', 'involving', 'bond', 'let', 'amwu', 'missiles', 'wages', 'ship',
                   'european', 'handled', 'recently', 'accord', 'camps', 'mistakes', 'employee', 'grenades', 'thousand',
                   'route', 'rubber', 'alternative', 'showing', 'camp', 'internal', 'tight', 'brazil', 'tourism',
                   'enterprise', 'hiding', 'subject', 'angry', 'critical', 'videotape', 'huegill', 'balmer',
                   'electricity', 'ministerial', 'acknowledged', 'palace', 'reaction', 'planned', 'broke', 'suspects',
                   'dramatic', 'jan', 'causing', 'intimidation', 'crashing', 'moussaoui', 'discuss', 'trouble',
                   'reportedly', 'failing', 'ending', 'standing', 'arthurs', 'macfarlane', 'finalised', 'care',
                   'progress', 'lennon', 'alan', 'asking', 'particularly', 'rising', 'timed', 'email', 'paper', 'sharp',
                   'casinos', 'safe', 'federation', 'fundamental', 'association', 'employed', 'spills', 'magistrate',
                   'mohamad', 'diagnosed', 'eligible', 'starts', 'supply', 'intervention', 'escude', 'trends', 'graham',
                   'carl', 'symbols', 'author', 'confessed', 'zaman', 'adopted', 'ran', 'genetically', 'nato', 'clash',
                   'saa', 'begins', 'turning', 'village', 'stefan', 'perpetrators', 'mainly', 'scouring', 'jirga',
                   'proves', 'shell', 'crowded', 'ranging', 'hear', 'trounson', 'robertson', 'advani', 'remember',
                   'upsurge',
                   'stumps', 'applications', 'urged', 'tougher', 'suggests', 'fans', 'formal', 'cope', 'templeton',
                   'counting', 'gabriel', 'slowing', 'powered', 'partnership', 'payments', 'travellers', 'villawood',
                   'rare', 'welcome', 'spin', 'destruction', 'thunderstorm', 'nautical', 'speculation', 'requests',
                   'faith', 'africans', 'pleased', 'ourselves', 'raise', 'musical', 'conspirators', 'incentive',
                   'upper', 'solvency', 'rioting', 'values', 'tommy', 'miles', 'withdrawal', 'initially', 'palmerston',
                   'downgrade', 'stance', 'arrivals', 'shifted', 'amazon', 'catch', 'equipment', 'tv', 'aviation',
                   'brussels', 'leak', 'kennedy', 'dick', 'chase', 'imagine', 'daily', 'wouldn', 'vowed', 'slips',
                   'hayward', 'sport', 'asian', 'hawke', 'gibbons', 'stoltenberg', 'guy', 'square', 'sinai', 'events',
                   'evening', 'dfat', 'olivier', 'monitored', 'franklin', 'model', 'feared', 'tailender', 'knife',
                   'tunnels', 'currency', 'likewise', 'britt', 'maintains', 'presidential', 'plants', 'importance',
                   'escalate', 'peru', 'housing', 'pervez', 'cairns', 'surprise', 'criticised', 'passenger', 'rfds',
                   'standards', 'solo', 'locations', 'nicorette', 'personally', 'towers', 'downturn', 'informed',
                   'dealing', 'langdale', 'wife', 'fought', 'arresting', 'regulation', 'luck', 'defunct', 'skippered',
                   'arab', 'assured', 'ferguson', 'reuters', 'wearing', 'perfect', 'cases', 'discussing', 'democracy',
                   'sheet', 'salaries', 'guards', 'undertaken', 'transitional', 'generally', 'tyco', 'abegglen',
                   'pitched', 'threw', 'maguire', 'score', 'institutions', 'emissions', 'broadcast', 'utn', 'wild',
                   'transferred', 'habeel', 'marked', 'eyes', 'protected', 'default', 'stephan', 'players', 'duck',
                   'satellite', 'considering', 'religious', 'lieutenant', 'mitsubishi', 'cover', 'declaring',
                   'hospitals', 'soil', 'belief', 'placing', 'device', 'shimon', 'grass', 'interested', 'chasing',
                   'stronger', 'announcement', 'bbc', 'perhaps', 'loya', 'rehman', 'alei', 'arriving', 'shaky', 'dewar',
                   'terrible', 'protesters', 'arabs', 'rescued', 'easy', 'everyone', 'reporters', 'owner',
                   'arbitration', 'iraq', 'terminal', 'stepping', 'visas', 'struggling', 'defendants', 'friend',
                   'advertising', 'decade', 'vessel', 'haitian', 'astronauts', 'shown', 'counter', 'battleground',
                   'girl', 'coastal', 'blast', 'development', 'evacuated', 'danger', 'contact', 'forecasting', 'saadi',
                   'infantry', 'consequences', 'unpredictable', 'lucky', 'offering', 'riots', 'recognised', 'memory',
                   'militias', 'silent', 'prohibited', 'operator', 'climate', 'comeback', 'communications', 'nasty',
                   'projections', 'combination', 'crack', 'ongoing', 'prix', 'export', 'sought', 'siege', 'attendants',
                   'nelson', 'legislation', 'jackson', 'occur', 'limits', 'alcohol', 'allowing', 'termination',
                   'strategy', 'unpaid', 'runway', 'casualties', 'ranked', 'date', 'firms', 'computer', 'enemy',
                   'timing', 'russell', 'demanded', 'islam', 'china', 'ashcroft', 'resigned', 'parliamentary',
                   'infrastructure', 'suburb', 'whilst', 'helen', 'bail', 'arm', 'funding', 'yunis', 'holds',
                   'relation', 'harm', 'waging', 'softer', 'bombed', 'proceed', 'earning', 'apply', 'deciding',
                   'living', 'corowa', 'prompted', 'canada', 'published', 'chemical', 'aziz', 'dollar', 'outsourcing',
                   'dialogue', 'extremely', 'ron', 'stray', 'beattie', 'gets', 'summer', 'behalf', 'computers',
                   'razzak', 'embryo', 'walker', 'policemen', 'globe', 'survival', 'ivf', 'type', 'approved', 'plant',
                   'split', 'restructuring', 'catches', 'recommendation', 'factors', 'interrogation', 'escalated',
                   'baker', 'wran', 'favour',
                   'gunman', 'returns', 'ideas', 'enormous', 'defeated', 'symptoms', 'blocks', 'compound', 'provincial',
                   'delayed', 'paktika', 'risen', 'points', 'fatality', 'obviously', 'champions', 'mounted', 'mosque',
                   'committed', 'music', 'surprised', 'boston', 'pr', 'negotiator', 'bradford', 'shop', 'profits',
                   'administrator', 'crimes', 'hawkesbury', 'vajpayee', 'personality', 'surrounded', 'resort', 'flee',
                   'swedish', 'acdt', 'join', 'fears', 'detained', 'sets', 'paceman', 'mount', 'todd', 'seem',
                   'arrival', 'myself', 'lali',
                   'judge', 'spain', 'defended', 'sweeping', 'hat', 'overrun', 'uncertain', 'junior', 'mountainous',
                   'previously', 'occurred', 'teams', 'example', 'pennsylvania', 'breathing', 'perkins', 'stores',
                   'totally', 'parents', 'teenager', 'ensuring', 'oval', 'happening', 'marks', 'vigil', 'aging',
                   'haven', 'june', 'fort', 'witness', 'bowled', 'christian', 'effectively', 'trials', 'england',
                   'option', 'sponsored', 'dying', 'prize', 'deadline', 'competitive', 'considered', 'controlled',
                   'exactly', 'maxi', 'honour', 'megawati', 'teenage', 'task', 'razor', 'consistent', 'resolved',
                   'canyon', 'ernst', 'bowlers', 'push', 'deserve', 'quit', 'contributions', 'eighth', 'heavily',
                   'appointment', 'protest', 'journey', 'card', 'lone', 'repeated', 'brother', 'ones', 'firemen',
                   'engines', 'interviewed', 'lew', 'uss', 'authorising', 'slip', 'command', 'causes', 'decline',
                   'collect', 'accountancy', 'acquitted', 'escape', 'principle', 'evil', 'priest', 'serve', 'linked',
                   'black', 'complacency', 'guide', 'funds', 'claiming', 'ntini', 'swimming', 'yassin', 'monetary',
                   'guarding', 'widespread', 'scarfe', 'aspects', 'presidency', 'miami', 'isolated', 'transfer',
                   'reasonably', 'impossible', 'fate', 'lindsay', 'obese', 'dickie', 'consent', 'augusta', 'poor',
                   'indicated', 'blamed', 'ruling', 'aimed', 'cheney', 'obesity', 'limited', 'winter', 'bayliss',
                   'learn', 'identify', 'decisive', 'congress', 'frank', 'midnight', 'hills', 'includes', 'wiget',
                   'neighbouring', 'setting', 'settler', 'democratic', 'reminded', 'johnston', 'article', 'talented',
                   'adult', 'once', 'taylor', 'reasons', 'sheldon', 'aim', 'peacekeeping', 'social', 'attended',
                   'stands', 'minor', 'masterminding', 'providing', 'remove', 'comply', 'meant', 'dominant', 'anybody',
                   'denies', 'raduyev', 'aiming', 'attempted', 'irrelevant', 'hectares', 'dawn', 'assist', 'appealed',
                   'profit', 'remote', 'commercial', 'love', 'indiana', 'sitting', 'coincide', 'swimmer', 'admits',
                   'adults', 'mounting', 'forests', 'blew', 'reputation', 'goes', 'shoalhaven', 'seeing', 'thick',
                   'deliberate', 'khan', 'striking', 'tribute', 'earth', 'principles', 'finishing', 'reading',
                   'christians', 'faced', 'embryos', 'strait', 'ali', 'panel', 'lunchtime', 'tower', 'defuse',
                   'securities', 'motor', 'paedophiles', 'managers', 'spill', 'worm', 'settlers', 'fly']
        self.assertTrue(a, True)
        self.assertEqual(wv1.index_to_key, ans_key)

    def test_ind2word(self):
        wv = readWordEmbedding('my_model1.bin')
        self.assertEqual(ind2word(wv,0),['the'])
        self.assertEqual(ind2word(wv, [0, 0, 0]),['the', 'the', 'the'])
        self.assertEqual(ind2word(wv,[0,1,2]),['the', 'to', 'of'])
        self.assertEqual(ind2word(wv,[3,6,8]),['in', 'is', 'on'])
        self.assertEqual(ind2word(wv, [378, 1226, 338,999,93,367]),['building', 'victorian', 'secretary', 'hold', 'very', 'winds'])

    def test_wordEmbeddingLayer(self):

        model = wordEncoding('test_wordEncoding.txt', MinCount=8, InitialLearnRate=0.05)
        layer = wordEmbeddingLayer(100, 8, Weights=model)
        a = False
        if isinstance(layer, my_layer):
            a = True
        self.assertTrue(a)

    def test_word2ind(self):
        wv = readWordEmbedding('my_model1.bin')
        self.assertEqual(word2ind(wv, 'the'), [0])
        self.assertEqual(word2ind(wv, ['the', 'the', 'the']), [0,0,0])
        self.assertEqual(word2ind(wv, ['the', 'to', 'of']), [0,1,2])
        self.assertEqual(word2ind(wv, ['in', 'is', 'on']), [3,6,8])
        self.assertEqual(word2ind(wv, ['building', 'victorian', 'secretary', 'hold', 'very', 'winds']),[378, 1226, 338, 999, 93, 367])

    def test_isVocabularyWord(self):
        wv = readWordEmbedding('my_model1.bin')
        self.assertEqual(isVocabularyWord(wv, 'king'), [1])
        self.assertEqual(isVocabularyWord(wv, ['the', 'fuck', 'the']), [1, 0, 1])
        self.assertEqual(isVocabularyWord(wv, ['the', 'to', 'of']), [1, 1, 1])
        self.assertEqual(isVocabularyWord(wv, ['in', 'is', 'on']), [1, 1, 1])
        self.assertEqual(isVocabularyWord(wv, ['emperor', 'victorian', 'secretary', 'hsaiu', 'very', 'guard']),[0, 1, 1, 0, 1, 1])

    def test_word2vec(self):
        wv = readWordEmbedding('my_model1.bin')
        a=word2vec(wv, 'king')
        c = word2vec(wv, 'queen')
        b = word2vec(wv, 'man')
        d = word2vec(wv, 'woman')
        d=a+d
        d=a+c-b+d
        e=a-b*c

    def test_vec2word(self):
        wv = readWordEmbedding('my_model1.bin')
        a = word2vec(wv, 'king')
        b = word2vec(wv, 'man')
        d = word2vec(wv, 'woman')
        e=a-b+d
        aa=vec2word(wv,a)
        bb=vec2word(wv,b)
        dd=vec2word(wv,d)
        ee=vec2word(wv,e)
        self.assertEqual(aa, 'king')
        self.assertEqual(ee, 'queen')
        self.assertEqual(bb, 'man')
        self.assertEqual(dd, 'woman')
        f=word2vec(wv, 'blow')+word2vec(wv, 'fire')+word2vec(wv, 'building')
        g=word2vec(wv, 'human')+word2vec(wv, 'war')+word2vec(wv, 'gun')
        ff = vec2word(wv, f)
        gg = vec2word(wv, g)
        self.assertEqual(ff, 'destroyed')
        self.assertEqual(gg, 'waging')

    def test_vec2word2(self):
        wv = readWordEmbedding('my_trained_model1.bin')
        a = word2vec(wv, 'king')
        b = word2vec(wv, 'man')
        d = word2vec(wv, 'woman')
        e=a-b+d
        aa=vec2word(wv,a,k=2)
        ee=vec2word(wv,e,k=5)
        self.assertEqual(aa, ['king', 'kings'])
        self.assertEqual(ee, ['king', 'queen', 'monarch', 'princess', 'prince'])

        f=word2vec(wv, 'water')+word2vec(wv, 'die')
        g=word2vec(wv, 'human')+word2vec(wv, 'wicked')-word2vec(wv, 'kind')
        ff = vec2word(wv, f,k=5)
        gg = vec2word(wv, g,k=3)
        self.assertEqual(ff, ['die', 'water', 'dying', 'drown', 'dies'])
        self.assertEqual(gg, ['wicked', 'human', 'evil'])
    def test_trainwordEmbedding(self):
        wv=trainWordEmbedding('test_wordEncoding.txt')
        wv1=trainWordEmbedding('test_wordEncoding.txt',MinCount=2)
        wv2 = trainWordEmbedding('test_wordEncoding.txt',NumNegativeSamples=5, ns_exponent=0.75, cbow_mean=1)
        wv3 = trainWordEmbedding('test_wordEncoding.txt',Dimension=100, InitialLearnRate=0.025, Window=5, MinCount=10)
        self.assertEqual(wv.index_to_key,['thou', 'thy', 'thee', 'self', 'thine', 'art', 'beauty', 'world', 'beautys', 'ten', 'still', 'doth', 'why', 'dost', 'another', 'sweet'])
        self.assertEqual(wv1.index_to_key, ['thou', 'thy', 'thee', 'self', 'thine', 'art', 'beautys', 'beauty', 'world', 'another', 'why', 'doth', 'still', 'ten', 'sweet', 'dost', 'make', 'love', 'eyes', 'time', 'age', 'shall', 'fair', 'shame', 'treasure', 'every', 'eye', 'single', 'live', 'look', 'glass', 'times', 'whose', 'own', 'die', 'sum', 'might', 'old', 'face', 'form', 'lies', 'child', 'praise', 'deep', 'winters', 'repair', 'else', 'tender', 'heir', 'waste', 'bear', 'fresh', 'worlds', 'lusty', 'change', 'mother', 'sweets', 'happier', 'shouldst', 'gracious', 'looks', 'resembling', 'like', 'thyself', 'music', 'joy', 'posterity', 'lovst', 'receivst', 'many', 'wilt', 'prove', 'none', 'widow', 'mind', 'place', 'ere', 'distilld', 'nor', 'calls', 'lovely', 'hate', 'shalt', 'golden', 'spend', 'beauteous', 'yet', 'canst', 'gone', 'unused', 'lives', 'gentle', 'summer', 'winter', 'confounds', 'left', 'desire'])
        self.assertEqual(wv2.index_to_key, ['thou', 'thy', 'thee', 'self', 'thine', 'art', 'beauty', 'world', 'beautys', 'ten', 'still', 'doth', 'why', 'dost', 'another', 'sweet'])
        self.assertEqual(wv3.index_to_key, ['thou', 'thy', 'thee', 'self', 'thine'])



# documents = [
#      "fairest creatures desire increase thereby beautys rose might never die riper time decease tender heir might bear memory thou contracted thine own bright eyes feedst thy lights flame selfsubstantial fuel making famine abundance lies thy self thy foe thy sweet self cruel thou art worlds fresh ornament herald gaudy spring thine own bud buriest thy content tender churl makst waste niggarding pity world else glutton eat worlds due grave thee",
#     "a a a a a a a a a a a forty winters shall besiege thy brow dig deep trenches thy beautys field thy youths proud livery gazed tatterd weed small worth held asked thy beauty lies treasure thy lusty days say thine own deep sunken eyes alleating shame thriftless praise praise deservd thy beautys thou couldst answer fair child mine shall sum count make old excuse proving beauty succession thine new made thou art old thy blood warm thou feelst cold",
#     "look thy glass tell face thou viewest time face form another whose fresh repair thou renewest thou dost beguile world unbless mother fair whose uneard womb disdains tillage thy husbandry fond tomb selflove stop posterity thou art thy mothers glass thee calls back lovely april prime thou windows thine age shalt despite wrinkles thy golden time thou live rememberd die single thine image dies thee",
#     "unthrifty loveliness why dost thou spend upon thy self thy beautys legacy natures bequest gives nothing doth lend frank lends free beauteous niggard why dost thou abuse bounteous largess thee give profitless usurer why dost thou great sum sums yet canst live traffic thy self alone thou thy self thy sweet self dost deceive nature calls thee gone acceptable audit canst thou leave thy unused beauty tombed thee lives th executor",
#     "hours gentle work frame lovely gaze every eye doth dwell play tyrants same unfair fairly doth excel neverresting time leads summer hideous winter confounds sap checked frost lusty leaves quite gone beauty oersnowed bareness every summers distillation left liquid prisoner pent walls glass beautys effect beauty bereft nor nor remembrance flowers distilld though winter meet leese show substance still lives sweet",
#     "let winters ragged hand deface thee thy summer ere thou distilld make sweet vial treasure thou place beautys treasure ere selfkilld forbidden usury happies pay willing loan thats thy self breed another thee ten times happier ten ten times thy self happier thou art ten thine ten times refigurd thee death thou shouldst depart leaving thee living posterity selfwilld thou art fair deaths conquest make worms thine heir",
#     "lo orient gracious light lifts up burning head eye doth homage newappearing sight serving looks sacred majesty climbd steepup heavenly hill resembling strong youth middle age yet mortal looks adore beauty still attending golden pilgrimage highmost pitch weary car like feeble age reeleth day eyes fore duteous converted low tract look another way thou thyself outgoing thy noon unlookd diest unless thou get son",
#     "music hear why hearst thou music sadly sweets sweets war joy delights joy why lovst thou thou receivst gladly else receivst pleasure thine annoy true concord welltuned sounds unions married offend thine ear sweetly chide thee confounds singleness parts thou shouldst bear mark string sweet husband another strikes mutual ordering resembling sire child happy mother pleasing note sing whose speechless song many seeming sings thee thou single wilt prove none",
#     "fear wet widows eye thou consumst thy self single life ah thou issueless shalt hap die world wail thee like makeless wife world thy widow still weep thou form thee hast left behind every private widow well keep childrens eyes husbands shape mind look unthrift world doth spend shifts place still world enjoys beautys waste hath world end kept unused user destroys love toward others bosom sits murdrous shame commits",
#     "shame deny thou bearst love thy self art unprovident grant thou wilt thou art belovd many thou none lovst evident thou art possessd murderous hate gainst thy self thou stickst conspire seeking beauteous roof ruinate repair thy chief desire o change thy thought change mind shall hate fairer lodgd gentle love thy presence gracious kind thyself least kindhearted prove make thee another self love beauty still live thine thee"
# ]
# #model =wordEncoding(documents)
#
# model =wordEncoding('test_wordEncoding.txt')
# model =wordEncoding('test_wordEncoding.txt',Dimension=100)
# model =wordEncoding('test_wordEncoding.txt',Window=10)
# model =wordEncoding('test_wordEncoding.txt',MinCount=8,InitialLearnRate=0.05)
#
# layer =wordEmbeddingLayer(100,8,Weights=model)
#
# print(model)
# vec_king = word2vec(model,'ten')
# print(vec_king)
# print(model.index_to_key)
# print('111')
# dictionary2=wordEncoding(documents)
# dictionary1=wordEncoding('test_wordEncoding.txt')
# print(dictionary1)
# print(dictionary1.token2id)
#
# new_doc=['thou art bud never art art','art art bud bud']
#
# new_vec = doc2sequence(dictionary1,new_doc)
# print(new_vec)
# wv=fastTextWordEmbedding()
# new_doc=['never emperor of the','the the of king queen']
# new_vec=doc2sequence(wv,new_doc)
# print(new_vec)
# new_vec=doc2sequence(wv,new_doc,'PaddingDirection','left')
# print(new_vec)
# new_doc=['never emperor of the','the the of king queen']
# new_vec=doc2sequence(wv,new_doc,'PaddingDirection','right','PaddingValue',66)
# print(new_vec)

# texts = [
#     [word for word in document.lower().split()]
#     for document in documents
# ]
# a = wordEncoding(texts)
# b=wordEncoding("test_wordEncoding.txt")
# print(a.token2id)
# print(a)



#
# wv=fastTextWordEmbedding()
# writeWordEmbedding(wv,"my_model2.bin")
#
# print(isVocabularyWord(wv,'of'))
# print(isVocabularyWord(wv,['of']))
# print(isVocabularyWord(wv,['of','the','emperor']))
# print(isVocabularyWord(wv,['queen','the','king']))
#
#
# print(word2ind(wv,'of'))
# print(word2ind(wv,['of']))
# print(word2ind(wv,['of','the','emperor']))
# print(word2ind(wv,['queen','the','king']))
#
# print(ind2word(wv,0))
# print(ind2word(wv,[0,1,2]))
# print(ind2word(wv,[3,6,8]))
#
# print(wv.key_to_index)
# print(wv.index_to_key)
# print(wv.vectors)
#
#
#
# vec_king=word2vec(wv,"king")
# vec_woman=word2vec(wv,"woman")
# vec_man=word2vec(wv,"man")
#
# vec_ans=vec_king-vec_man+vec_woman
# my_ans=vec2word(wv,vec_king+vec_woman-vec_man)
#
# print(my_ans)


if __name__ == '__main__':
    unittest.main()
