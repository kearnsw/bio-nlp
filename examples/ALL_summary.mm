{"AllDocuments":[
{
     "Document": {
         "CmdLine": {
             "Command": "metamap16.BINARY.Linux --lexicon db -Z 2016AA --JSONf 4",
             "Options": [
                 {
                     "OptName": "lexicon",
                     "OptValue": "db"
                 },
                 {
                     "OptName": "mm_data_year",
                     "OptValue": "2016AA"
                 },
                 {
                     "OptName": "JSONf",
                     "OptValue": "4"
                 },
                 {
                     "OptName": "infile",
                     "OptValue": "user_input"
                 },
                 {
                     "OptName": "outfile",
                     "OptValue": "user_output"
                 }]
         },
         "AAs": [
             {
                 "AAText": "ALL",
                 "AAExp": "acute lymphocytic leukemia",
                 "AATokenNum": "1",
                 "AALen": "3",
                 "AAExpTokenNum": "5",
                 "AAExpLen": "26",
                 "AAStartPos": "332",
                 "AACUIs": ["C1961102"]
             }],
         "Negations": [],
         "Utterances": [
             {
                 "PMID": "00000000",
                 "UttSection": "tx",
                 "UttNum": "1",
                 "UttText": "Leukemia is cancer of the white blood cells. ",
                 "UttStartPos": "0",
                 "UttLength": "45",
                 "Phrases": [
                     {
                         "PhraseText": "Leukemia",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "leukemia",
                                 "InputMatch": "Leukemia",
                                 "LexCat": "noun",
                                 "Tokens": ["leukemia"]
                             }],
                         "PhraseStartPos": "0",
                         "PhraseLength": "8",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0023418",
                                         "CandidateMatched": "LEUKAEMIA",
                                         "CandidatePreferred": "leukemia",
                                         "MatchedWords": ["leukaemia"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","ICPC","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_CTEP-SDC","NCI_NCI-GLOSS","NCI_NICHD","NDFRT","OMIM","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "0",
                                                 "Length": "8"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "is",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "aux",
                                 "LexMatch": "is",
                                 "InputMatch": "is",
                                 "LexCat": "aux",
                                 "Tokens": ["is"]
                             }],
                         "PhraseStartPos": "9",
                         "PhraseLength": "2",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "cancer of the white blood cells.",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "cancer",
                                 "InputMatch": "cancer",
                                 "LexCat": "noun",
                                 "Tokens": ["cancer"]
                             },
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "of",
                                 "InputMatch": "of",
                                 "LexCat": "prep",
                                 "Tokens": ["of"]
                             },
                             {
                                 "SyntaxType": "det",
                                 "LexMatch": "the",
                                 "InputMatch": "the",
                                 "LexCat": "det",
                                 "Tokens": ["the"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "white blood cells",
                                 "InputMatch": "white blood cells",
                                 "LexCat": "noun",
                                 "Tokens": ["white","blood","cells"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ".",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "12",
                         "PhraseLength": "32",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-780",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0006826",
                                         "CandidateMatched": "CANCER",
                                         "CandidatePreferred": "Malignant Neoplasms",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AIR","AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_FDA","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "12",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-666",
                                         "CandidateCUI": "C0023508",
                                         "CandidateMatched": "White Blood Cells",
                                         "CandidatePreferred": "White Blood Cell Count procedure",
                                         "MatchedWords": ["white","blood","cells"],
                                         "SemTypes": ["lbpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","MSH","MTH","NCI","NCI_CDISC","NLMSubSyn","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "26",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-780",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0006826",
                                         "CandidateMatched": "CANCER",
                                         "CandidatePreferred": "Malignant Neoplasms",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AIR","AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_FDA","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "12",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-666",
                                         "CandidateCUI": "C0023516",
                                         "CandidateMatched": "White Blood Cells",
                                         "CandidatePreferred": "Leukocytes",
                                         "MatchedWords": ["white","blood","cells"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","LCH_NW","LNC","MSH","MTH","NCI","NCI_NCI-GLOSS","NLMSubSyn","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "26",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-780",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0998265",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Cancer Genus",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["euka"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCBI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "12",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-666",
                                         "CandidateCUI": "C0023508",
                                         "CandidateMatched": "White Blood Cells",
                                         "CandidatePreferred": "White Blood Cell Count procedure",
                                         "MatchedWords": ["white","blood","cells"],
                                         "SemTypes": ["lbpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","MSH","MTH","NCI","NCI_CDISC","NLMSubSyn","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "26",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-780",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0998265",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Cancer Genus",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["euka"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCBI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "12",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-666",
                                         "CandidateCUI": "C0023516",
                                         "CandidateMatched": "White Blood Cells",
                                         "CandidatePreferred": "Leukocytes",
                                         "MatchedWords": ["white","blood","cells"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","LCH_NW","LNC","MSH","MTH","NCI","NCI_NCI-GLOSS","NLMSubSyn","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "26",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-780",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1306459",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Primary malignant neoplasm",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "12",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-666",
                                         "CandidateCUI": "C0023508",
                                         "CandidateMatched": "White Blood Cells",
                                         "CandidatePreferred": "White Blood Cell Count procedure",
                                         "MatchedWords": ["white","blood","cells"],
                                         "SemTypes": ["lbpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","MSH","MTH","NCI","NCI_CDISC","NLMSubSyn","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "26",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-780",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1306459",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Primary malignant neoplasm",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "12",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-666",
                                         "CandidateCUI": "C0023516",
                                         "CandidateMatched": "White Blood Cells",
                                         "CandidatePreferred": "Leukocytes",
                                         "MatchedWords": ["white","blood","cells"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","LCH_NW","LNC","MSH","MTH","NCI","NCI_NCI-GLOSS","NLMSubSyn","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "26",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     }]
             },
             {
                 "PMID": "00000000",
                 "UttSection": "tx",
                 "UttNum": "2",
                 "UttText": "White blood cells help your body fight infection. ",
                 "UttStartPos": "45",
                 "UttLength": "50",
                 "Phrases": [
                     {
                         "PhraseText": "White blood cells",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "white blood cells",
                                 "InputMatch": "White blood cells",
                                 "LexCat": "noun",
                                 "Tokens": ["white","blood","cells"]
                             }],
                         "PhraseStartPos": "45",
                         "PhraseLength": "17",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0023508",
                                         "CandidateMatched": "White Blood Cells",
                                         "CandidatePreferred": "White Blood Cell Count procedure",
                                         "MatchedWords": ["white","blood","cells"],
                                         "SemTypes": ["lbpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","MSH","MTH","NCI","NCI_CDISC","NLMSubSyn","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "45",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0023516",
                                         "CandidateMatched": "White Blood Cells",
                                         "CandidatePreferred": "Leukocytes",
                                         "MatchedWords": ["white","blood","cells"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","LCH_NW","LNC","MSH","MTH","NCI","NCI_NCI-GLOSS","NLMSubSyn","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "45",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "help",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "verb",
                                 "LexMatch": "help",
                                 "InputMatch": "help",
                                 "LexCat": "verb",
                                 "Tokens": ["help"]
                             }],
                         "PhraseStartPos": "63",
                         "PhraseLength": "4",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C1269765",
                                         "CandidateMatched": "Help",
                                         "CandidatePreferred": "Assisted (qualifier value)",
                                         "MatchedWords": ["help"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "63",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C1552861",
                                         "CandidateMatched": "help",
                                         "CandidatePreferred": "Help document",
                                         "MatchedWords": ["help"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V3.0","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "63",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "your body fight infection.",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "pron",
                                 "LexMatch": "your",
                                 "InputMatch": "your",
                                 "LexCat": "pron",
                                 "Tokens": ["your"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "body",
                                 "InputMatch": "body",
                                 "LexCat": "noun",
                                 "Tokens": ["body"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "fight",
                                 "InputMatch": "fight",
                                 "LexCat": "noun",
                                 "Tokens": ["fight"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "infection",
                                 "InputMatch": "infection",
                                 "LexCat": "noun",
                                 "Tokens": ["infection"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ".",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "68",
                         "PhraseLength": "26",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-851",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0242821",
                                         "CandidateMatched": "body",
                                         "CandidatePreferred": "Human body",
                                         "MatchedWords": ["body"],
                                         "SemTypes": ["humn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LCH","LCH_NW","MSH","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "73",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0424324",
                                         "CandidateMatched": "fight",
                                         "CandidatePreferred": "Fighting",
                                         "MatchedWords": ["fight"],
                                         "SemTypes": ["socb"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "78",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-827",
                                         "CandidateCUI": "C0009450",
                                         "CandidateMatched": "Infection, NOS",
                                         "CandidatePreferred": "Communicable Diseases",
                                         "MatchedWords": ["infection"],
                                         "SemTypes": ["dsyn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","COSTAR","CSP","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_NICHD","NDFRT","NLMSubSyn","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "84",
                                                 "Length": "9"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-851",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0242821",
                                         "CandidateMatched": "body",
                                         "CandidatePreferred": "Human body",
                                         "MatchedWords": ["body"],
                                         "SemTypes": ["humn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LCH","LCH_NW","MSH","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "73",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0424324",
                                         "CandidateMatched": "fight",
                                         "CandidatePreferred": "Fighting",
                                         "MatchedWords": ["fight"],
                                         "SemTypes": ["socb"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "78",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-827",
                                         "CandidateCUI": "C3714514",
                                         "CandidateMatched": "INFECTION",
                                         "CandidatePreferred": "Infection",
                                         "MatchedWords": ["infection"],
                                         "SemTypes": ["patf"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","COSTAR","CST","DXP","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","NCI_NCI-GLOSS","PDQ","SNM","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "84",
                                                 "Length": "9"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-851",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0460148",
                                         "CandidateMatched": "Body",
                                         "CandidatePreferred": "Human body structure",
                                         "MatchedWords": ["body"],
                                         "SemTypes": ["anst"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","FMA","MTH","NLMSubSyn","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "73",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0424324",
                                         "CandidateMatched": "fight",
                                         "CandidatePreferred": "Fighting",
                                         "MatchedWords": ["fight"],
                                         "SemTypes": ["socb"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "78",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-827",
                                         "CandidateCUI": "C0009450",
                                         "CandidateMatched": "Infection, NOS",
                                         "CandidatePreferred": "Communicable Diseases",
                                         "MatchedWords": ["infection"],
                                         "SemTypes": ["dsyn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","COSTAR","CSP","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_NICHD","NDFRT","NLMSubSyn","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "84",
                                                 "Length": "9"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-851",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0460148",
                                         "CandidateMatched": "Body",
                                         "CandidatePreferred": "Human body structure",
                                         "MatchedWords": ["body"],
                                         "SemTypes": ["anst"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","FMA","MTH","NLMSubSyn","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "73",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0424324",
                                         "CandidateMatched": "fight",
                                         "CandidatePreferred": "Fighting",
                                         "MatchedWords": ["fight"],
                                         "SemTypes": ["socb"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "78",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-827",
                                         "CandidateCUI": "C3714514",
                                         "CandidateMatched": "INFECTION",
                                         "CandidatePreferred": "Infection",
                                         "MatchedWords": ["infection"],
                                         "SemTypes": ["patf"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","COSTAR","CST","DXP","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","NCI_NCI-GLOSS","PDQ","SNM","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "84",
                                                 "Length": "9"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-851",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C1268086",
                                         "CandidateMatched": "Body",
                                         "CandidatePreferred": "Body structure",
                                         "MatchedWords": ["body"],
                                         "SemTypes": ["anst"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CDISC","NCI_NICHD","SNOMEDCT_US","SNOMEDCT_VET"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "73",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0424324",
                                         "CandidateMatched": "fight",
                                         "CandidatePreferred": "Fighting",
                                         "MatchedWords": ["fight"],
                                         "SemTypes": ["socb"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "78",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-827",
                                         "CandidateCUI": "C0009450",
                                         "CandidateMatched": "Infection, NOS",
                                         "CandidatePreferred": "Communicable Diseases",
                                         "MatchedWords": ["infection"],
                                         "SemTypes": ["dsyn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","COSTAR","CSP","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_NICHD","NDFRT","NLMSubSyn","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "84",
                                                 "Length": "9"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-851",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C1268086",
                                         "CandidateMatched": "Body",
                                         "CandidatePreferred": "Body structure",
                                         "MatchedWords": ["body"],
                                         "SemTypes": ["anst"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CDISC","NCI_NICHD","SNOMEDCT_US","SNOMEDCT_VET"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "73",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C0424324",
                                         "CandidateMatched": "fight",
                                         "CandidatePreferred": "Fighting",
                                         "MatchedWords": ["fight"],
                                         "SemTypes": ["socb"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "78",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-827",
                                         "CandidateCUI": "C3714514",
                                         "CandidateMatched": "INFECTION",
                                         "CandidatePreferred": "Infection",
                                         "MatchedWords": ["infection"],
                                         "SemTypes": ["patf"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","COSTAR","CST","DXP","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","NCI_NCI-GLOSS","PDQ","SNM","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "84",
                                                 "Length": "9"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     }]
             },
             {
                 "PMID": "00000000",
                 "UttSection": "tx",
                 "UttNum": "3",
                 "UttText": "Your blood cells form in your bone marrow. ",
                 "UttStartPos": "95",
                 "UttLength": "43",
                 "Phrases": [
                     {
                         "PhraseText": "Your blood cells",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "pron",
                                 "LexMatch": "your",
                                 "InputMatch": "Your",
                                 "LexCat": "pron",
                                 "Tokens": ["your"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "blood cells",
                                 "InputMatch": "blood cells",
                                 "LexCat": "noun",
                                 "Tokens": ["blood","cells"]
                             }],
                         "PhraseStartPos": "95",
                         "PhraseLength": "16",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0005773",
                                         "CandidateMatched": "Blood Cells",
                                         "CandidatePreferred": "Blood Cells",
                                         "MatchedWords": ["blood","cells"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","LCH","LCH_NW","MEDLINEPLUS","MSH","MTH","NCI","NLMSubSyn","SNM","SNMI","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "100",
                                                 "Length": "11"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "form in your bone marrow.",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "verb",
                                 "LexMatch": "form",
                                 "InputMatch": "form",
                                 "LexCat": "verb",
                                 "Tokens": ["form"]
                             },
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "in",
                                 "InputMatch": "in",
                                 "LexCat": "prep",
                                 "Tokens": ["in"]
                             },
                             {
                                 "SyntaxType": "pron",
                                 "LexMatch": "your",
                                 "InputMatch": "your",
                                 "LexCat": "pron",
                                 "Tokens": ["your"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "bone marrow",
                                 "InputMatch": "bone marrow",
                                 "LexCat": "noun",
                                 "Tokens": ["bone","marrow"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ".",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "112",
                         "PhraseLength": "25",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-745",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-797",
                                         "CandidateCUI": "C1286272",
                                         "CandidateMatched": "bone form",
                                         "CandidatePreferred": "Form of bone",
                                         "MatchedWords": ["bone","form"],
                                         "SemTypes": ["clna"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             },
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "2",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "112",
                                                 "Length": "4"
                                             },
                                             {
                                                 "StartPos": "125",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-760",
                                         "CandidateCUI": "C0086590",
                                         "CandidateMatched": "Marrow",
                                         "CandidatePreferred": "Vegetable marrow",
                                         "MatchedWords": ["marrow"],
                                         "SemTypes": ["food"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "5",
                                                 "TextMatchEnd": "5",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["LCH","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "130",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-745",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-797",
                                         "CandidateCUI": "C1286272",
                                         "CandidateMatched": "bone form",
                                         "CandidatePreferred": "Form of bone",
                                         "MatchedWords": ["bone","form"],
                                         "SemTypes": ["clna"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             },
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "2",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "112",
                                                 "Length": "4"
                                             },
                                             {
                                                 "StartPos": "125",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-760",
                                         "CandidateCUI": "C0376152",
                                         "CandidateMatched": "Marrow",
                                         "CandidatePreferred": "Marrow",
                                         "MatchedWords": ["marrow"],
                                         "SemTypes": ["bpoc"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "5",
                                                 "TextMatchEnd": "5",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","HL7V2.5","MSH","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "130",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-745",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-797",
                                         "CandidateCUI": "C1286272",
                                         "CandidateMatched": "bone form",
                                         "CandidatePreferred": "Form of bone",
                                         "MatchedWords": ["bone","form"],
                                         "SemTypes": ["clna"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             },
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "2",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "112",
                                                 "Length": "4"
                                             },
                                             {
                                                 "StartPos": "125",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-760",
                                         "CandidateCUI": "C1546708",
                                         "CandidateMatched": "Marrow",
                                         "CandidatePreferred": "Marrow - Specimen Source Codes",
                                         "MatchedWords": ["marrow"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "5",
                                                 "TextMatchEnd": "5",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "130",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-745",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-760",
                                         "CandidateCUI": "C0348078",
                                         "CandidateMatched": "Form",
                                         "CandidatePreferred": "Qualitative form",
                                         "MatchedWords": ["form"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "112",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-806",
                                         "CandidateCUI": "C0005953",
                                         "CandidateMatched": "BONE MARROW",
                                         "CandidatePreferred": "Bone Marrow",
                                         "MatchedWords": ["bone","marrow"],
                                         "SemTypes": ["tisu"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "5",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","ICF","ICF-CY","LCH_NW","LNC","MSH","MTH","NCI","NCI_CDISC","NCI_NCI-GLOSS","NCI_NICHD","SNM","SNMI","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "125",
                                                 "Length": "11"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-745",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-760",
                                         "CandidateCUI": "C0376315",
                                         "CandidateMatched": "Form",
                                         "CandidatePreferred": "Manufactured form",
                                         "MatchedWords": ["form"],
                                         "SemTypes": ["mnob"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["LNC","MSH","MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "112",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-806",
                                         "CandidateCUI": "C0005953",
                                         "CandidateMatched": "BONE MARROW",
                                         "CandidatePreferred": "Bone Marrow",
                                         "MatchedWords": ["bone","marrow"],
                                         "SemTypes": ["tisu"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "5",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","ICF","ICF-CY","LCH_NW","LNC","MSH","MTH","NCI","NCI_CDISC","NCI_NCI-GLOSS","NCI_NICHD","SNM","SNMI","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "125",
                                                 "Length": "11"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-745",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-760",
                                         "CandidateCUI": "C1522492",
                                         "CandidateMatched": "Form",
                                         "CandidatePreferred": "Formation",
                                         "MatchedWords": ["form"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "112",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-806",
                                         "CandidateCUI": "C0005953",
                                         "CandidateMatched": "BONE MARROW",
                                         "CandidatePreferred": "Bone Marrow",
                                         "MatchedWords": ["bone","marrow"],
                                         "SemTypes": ["tisu"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "5",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","ICF","ICF-CY","LCH_NW","LNC","MSH","MTH","NCI","NCI_CDISC","NCI_NCI-GLOSS","NCI_NICHD","SNM","SNMI","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "125",
                                                 "Length": "11"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     }]
             },
             {
                 "PMID": "00000000",
                 "UttSection": "tx",
                 "UttNum": "4",
                 "UttText": "In leukemia, however, the bone marrow produces abnormal white blood cells. ",
                 "UttStartPos": "138",
                 "UttLength": "75",
                 "Phrases": [
                     {
                         "PhraseText": "In leukemia,",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "in",
                                 "InputMatch": "In",
                                 "LexCat": "prep",
                                 "Tokens": ["in"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "leukemia",
                                 "InputMatch": "leukemia",
                                 "LexCat": "noun",
                                 "Tokens": ["leukemia"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ",",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "138",
                         "PhraseLength": "12",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0023418",
                                         "CandidateMatched": "LEUKAEMIA",
                                         "CandidatePreferred": "leukemia",
                                         "MatchedWords": ["leukaemia"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","ICPC","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_CTEP-SDC","NCI_NCI-GLOSS","NCI_NICHD","NDFRT","OMIM","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "141",
                                                 "Length": "8"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "however,",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "adv",
                                 "LexMatch": "however",
                                 "InputMatch": "however",
                                 "LexCat": "adv",
                                 "Tokens": ["however"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ",",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "151",
                         "PhraseLength": "8",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "the bone marrow",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "det",
                                 "LexMatch": "the",
                                 "InputMatch": "the",
                                 "LexCat": "det",
                                 "Tokens": ["the"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "bone marrow",
                                 "InputMatch": "bone marrow",
                                 "LexCat": "noun",
                                 "Tokens": ["bone","marrow"]
                             }],
                         "PhraseStartPos": "160",
                         "PhraseLength": "15",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0005953",
                                         "CandidateMatched": "BONE MARROW",
                                         "CandidatePreferred": "Bone Marrow",
                                         "MatchedWords": ["bone","marrow"],
                                         "SemTypes": ["tisu"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","ICF","ICF-CY","LCH_NW","LNC","MSH","MTH","NCI","NCI_CDISC","NCI_NCI-GLOSS","NCI_NICHD","SNM","SNMI","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "164",
                                                 "Length": "11"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "produces",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "verb",
                                 "LexMatch": "produces",
                                 "InputMatch": "produces",
                                 "LexCat": "verb",
                                 "Tokens": ["produces"]
                             }],
                         "PhraseStartPos": "176",
                         "PhraseLength": "8",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "abnormal white blood cells.",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "abnormal",
                                 "InputMatch": "abnormal",
                                 "LexCat": "adj",
                                 "Tokens": ["abnormal"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "white blood cells",
                                 "InputMatch": "white blood cells",
                                 "LexCat": "noun",
                                 "Tokens": ["white","blood","cells"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ".",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "185",
                         "PhraseLength": "27",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0152009",
                                         "CandidateMatched": "ABNORMAL WHITE BLOOD CELLS",
                                         "CandidatePreferred": "White blood cell abnormality",
                                         "MatchedWords": ["abnormal","white","blood","cells"],
                                         "SemTypes": ["fndg"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             },
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "2",
                                                 "ConcMatchEnd": "4",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","CST","HPO","ICPC","LNC","NLMSubSyn","OMIM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "185",
                                                 "Length": "26"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     }]
             },
             {
                 "PMID": "00000000",
                 "UttSection": "tx",
                 "UttNum": "5",
                 "UttText": "These cells crowd out the healthy blood cells, making it hard for blood to do its work. ",
                 "UttStartPos": "213",
                 "UttLength": "88",
                 "Phrases": [
                     {
                         "PhraseText": "These cells",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "det",
                                 "LexMatch": "these",
                                 "InputMatch": "These",
                                 "LexCat": "det",
                                 "Tokens": ["these"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "cells",
                                 "InputMatch": "cells",
                                 "LexCat": "noun",
                                 "Tokens": ["cells"]
                             }],
                         "PhraseStartPos": "213",
                         "PhraseLength": "11",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0007634",
                                         "CandidateMatched": "Cells",
                                         "CandidatePreferred": "Cells",
                                         "MatchedWords": ["cells"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","CSP","FMA","GO","LCH","LCH_NW","LNC","MSH","MTH","NCI","NCI_NCI-GLOSS","NCI_UCUM","SNM","SNMI","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "219",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C3282337",
                                         "CandidateMatched": "Cells",
                                         "CandidatePreferred": "Cells [Chemical/Ingredient]",
                                         "MatchedWords": ["cells"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NDFRT"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "219",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "crowd",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "verb",
                                 "LexMatch": "crowd",
                                 "InputMatch": "crowd",
                                 "LexCat": "verb",
                                 "Tokens": ["crowd"]
                             }],
                         "PhraseStartPos": "225",
                         "PhraseLength": "5",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0010383",
                                         "CandidateMatched": "Crowd",
                                         "CandidatePreferred": "Crowding",
                                         "MatchedWords": ["crowd"],
                                         "SemTypes": ["socb"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","CSP","LCH_NW","LNC","MSH","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "225",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "out",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "adv",
                                 "LexMatch": "out",
                                 "InputMatch": "out",
                                 "LexCat": "adv",
                                 "Tokens": ["out"]
                             }],
                         "PhraseStartPos": "231",
                         "PhraseLength": "3",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0439787",
                                         "CandidateMatched": "Out",
                                         "CandidatePreferred": "Out (direction)",
                                         "MatchedWords": ["out"],
                                         "SemTypes": ["spco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "231",
                                                 "Length": "3"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0849355",
                                         "CandidateMatched": "Out",
                                         "CandidatePreferred": "Removed",
                                         "MatchedWords": ["out"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "231",
                                                 "Length": "3"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "the healthy blood cells,",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "det",
                                 "LexMatch": "the",
                                 "InputMatch": "the",
                                 "LexCat": "det",
                                 "Tokens": ["the"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "healthy",
                                 "InputMatch": "healthy",
                                 "LexCat": "adj",
                                 "Tokens": ["healthy"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "blood cells",
                                 "InputMatch": "blood cells",
                                 "LexCat": "noun",
                                 "Tokens": ["blood","cells"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ",",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "235",
                         "PhraseLength": "24",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-901",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-660",
                                         "CandidateCUI": "C3898900",
                                         "CandidateMatched": "Healthy",
                                         "CandidatePreferred": "Healthy",
                                         "MatchedWords": ["healthy"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "239",
                                                 "Length": "7"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-901",
                                         "CandidateCUI": "C0005773",
                                         "CandidateMatched": "Blood Cells",
                                         "CandidatePreferred": "Blood Cells",
                                         "MatchedWords": ["blood","cells"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","LCH","LCH_NW","MEDLINEPLUS","MSH","MTH","NCI","NLMSubSyn","SNM","SNMI","SNOMEDCT_US","UWDA"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "247",
                                                 "Length": "11"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "making",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "verb",
                                 "LexMatch": "making",
                                 "InputMatch": "making",
                                 "LexCat": "verb",
                                 "Tokens": ["making"]
                             }],
                         "PhraseStartPos": "260",
                         "PhraseLength": "6",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-966",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-966",
                                         "CandidateCUI": "C1881534",
                                         "CandidateMatched": "Make",
                                         "CandidatePreferred": "Make - Instruction Imperative",
                                         "MatchedWords": ["make"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "1"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "260",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "it hard for blood to do",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "pron",
                                 "LexMatch": "it",
                                 "InputMatch": "it",
                                 "LexCat": "pron",
                                 "Tokens": ["it"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "hard",
                                 "InputMatch": "hard",
                                 "LexCat": "adj",
                                 "Tokens": ["hard"]
                             },
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "for",
                                 "InputMatch": "for",
                                 "LexCat": "prep",
                                 "Tokens": ["for"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "blood",
                                 "InputMatch": "blood",
                                 "LexCat": "noun",
                                 "Tokens": ["blood"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "to do",
                                 "InputMatch": "to do",
                                 "LexCat": "noun",
                                 "Tokens": ["to","do"]
                             }],
                         "PhraseStartPos": "267",
                         "PhraseLength": "23",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-742",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-742",
                                         "CandidateCUI": "C0425659",
                                         "CandidateMatched": "Hard blood vessel",
                                         "CandidatePreferred": "Hard blood vessel",
                                         "MatchedWords": ["hard","blood","vessel"],
                                         "SemTypes": ["fndg"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             },
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "2",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "4"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "270",
                                                 "Length": "4"
                                             },
                                             {
                                                 "StartPos": "279",
                                                 "Length": "5"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "its work.",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "pron",
                                 "LexMatch": "its",
                                 "InputMatch": "its",
                                 "LexCat": "pron",
                                 "Tokens": ["its"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "work",
                                 "InputMatch": "work",
                                 "LexCat": "noun",
                                 "Tokens": ["work"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ".",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "291",
                         "PhraseLength": "9",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0043227",
                                         "CandidateMatched": "Work",
                                         "CandidatePreferred": "Work",
                                         "MatchedWords": ["work"],
                                         "SemTypes": ["ocac"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","CSP","LCH","LCH_NW","LNC","MSH","MTH","NCI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "295",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     }]
             },
             {
                 "PMID": "00000000",
                 "UttSection": "tx",
                 "UttNum": "6",
                 "UttText": "In acute lymphocytic leukemia (ALL), also called acute lymphoblastic leukemia, there are too many of specific types of white blood cells called lymphocytes or lymphoblasts. ",
                 "UttStartPos": "301",
                 "UttLength": "173",
                 "Phrases": [
                     {
                         "PhraseText": "In acute lymphocytic leukemia (ALL),",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "in",
                                 "InputMatch": "In",
                                 "LexCat": "prep",
                                 "Tokens": ["in"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "acute lymphocytic leukemia",
                                 "InputMatch": "acute lymphocytic leukemia",
                                 "LexCat": "noun",
                                 "Tokens": ["acute","lymphocytic","leukemia"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ",",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "301",
                         "PhraseLength": "36",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0023449",
                                         "CandidateMatched": "Acute Lymphocytic Leukaemia",
                                         "CandidatePreferred": "Acute lymphocytic leukemia",
                                         "MatchedWords": ["acute","lymphocytic","leukaemia"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","COSTAR","CSP","CST","HPO","ICD10CM","ICD9CM","MEDLINEPLUS","MTH","NCI","NCI_CDISC","NCI_CTEP-SDC","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","OMIM","PDQ","QMR","SNM","SNOMEDCT_US","SNOMEDCT_VET"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "304",
                                                 "Length": "26"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C1961102",
                                         "CandidateMatched": "Acute Lymphocytic Leukemia",
                                         "CandidatePreferred": "Precursor Cell Lymphoblastic Leukemia Lymphoma",
                                         "MatchedWords": ["acute","lymphocytic","leukemia"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["LCH_NW","MEDLINEPLUS","MSH","MTH","NDFRT","NLMSubSyn","OMIM","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "304",
                                                 "Length": "26"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "also",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "adv",
                                 "LexMatch": "also",
                                 "InputMatch": "also",
                                 "LexCat": "adv",
                                 "Tokens": ["also"]
                             }],
                         "PhraseStartPos": "338",
                         "PhraseLength": "4",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "called",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "verb",
                                 "LexMatch": "called",
                                 "InputMatch": "called",
                                 "LexCat": "verb",
                                 "Tokens": ["called"]
                             }],
                         "PhraseStartPos": "343",
                         "PhraseLength": "6",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-966",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-966",
                                         "CandidateCUI": "C0679006",
                                         "CandidateMatched": "Call",
                                         "CandidatePreferred": "Decision",
                                         "MatchedWords": ["call"],
                                         "SemTypes": ["menp"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "1"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "343",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-966",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-966",
                                         "CandidateCUI": "C1947967",
                                         "CandidateMatched": "Call",
                                         "CandidatePreferred": "Call (Instruction)",
                                         "MatchedWords": ["call"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "1"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "343",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "acute lymphoblastic leukemia,",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "acute lymphoblastic leukemia",
                                 "InputMatch": "acute lymphoblastic leukemia",
                                 "LexCat": "noun",
                                 "Tokens": ["acute","lymphoblastic","leukemia"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ",",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "350",
                         "PhraseLength": "29",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0023449",
                                         "CandidateMatched": "Acute lymphoblastic leukaemia",
                                         "CandidatePreferred": "Acute lymphocytic leukemia",
                                         "MatchedWords": ["acute","lymphoblastic","leukaemia"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","COSTAR","CSP","CST","HPO","ICD10CM","ICD9CM","MEDLINEPLUS","MTH","NCI","NCI_CDISC","NCI_CTEP-SDC","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","OMIM","PDQ","QMR","SNM","SNOMEDCT_US","SNOMEDCT_VET"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "350",
                                                 "Length": "28"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C1961102",
                                         "CandidateMatched": "Acute Lymphoblastic Leukemia",
                                         "CandidatePreferred": "Precursor Cell Lymphoblastic Leukemia Lymphoma",
                                         "MatchedWords": ["acute","lymphoblastic","leukemia"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["LCH_NW","MEDLINEPLUS","MSH","MTH","NDFRT","NLMSubSyn","OMIM","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "350",
                                                 "Length": "28"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C3542401",
                                         "CandidateMatched": "Acute Lymphoblastic Leukemia",
                                         "CandidatePreferred": "NCI CTEP SDC Acute Lymphoblastic Leukemia Sub-Category Terminology",
                                         "MatchedWords": ["acute","lymphoblastic","leukemia"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI","NCI_CTEP-SDC","NLMSubSyn"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "350",
                                                 "Length": "28"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "there",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "adv",
                                 "LexMatch": "there",
                                 "InputMatch": "there",
                                 "LexCat": "adv",
                                 "Tokens": ["there"]
                             }],
                         "PhraseStartPos": "380",
                         "PhraseLength": "5",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "are",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "aux",
                                 "LexMatch": "are",
                                 "InputMatch": "are",
                                 "LexCat": "aux",
                                 "Tokens": ["are"]
                             }],
                         "PhraseStartPos": "386",
                         "PhraseLength": "3",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "too",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "adv",
                                 "LexMatch": "too",
                                 "InputMatch": "too",
                                 "LexCat": "adv",
                                 "Tokens": ["too"]
                             }],
                         "PhraseStartPos": "390",
                         "PhraseLength": "3",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "many of specific types of white blood cells",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "det",
                                 "LexMatch": "many",
                                 "InputMatch": "many",
                                 "LexCat": "det",
                                 "Tokens": ["many"]
                             },
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "of",
                                 "InputMatch": "of",
                                 "LexCat": "prep",
                                 "Tokens": ["of"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "specific",
                                 "InputMatch": "specific",
                                 "LexCat": "adj",
                                 "Tokens": ["specific"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "types",
                                 "InputMatch": "types",
                                 "LexCat": "noun",
                                 "Tokens": ["types"]
                             },
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "of",
                                 "InputMatch": "of",
                                 "LexCat": "prep",
                                 "Tokens": ["of"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "white blood cells",
                                 "InputMatch": "white blood cells",
                                 "LexCat": "noun",
                                 "Tokens": ["white","blood","cells"]
                             }],
                         "PhraseStartPos": "394",
                         "PhraseLength": "43",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-769",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-744",
                                         "CandidateCUI": "C0205369",
                                         "CandidateMatched": "Specific",
                                         "CandidatePreferred": "Specific qualifier value",
                                         "MatchedWords": ["specific"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "402",
                                                 "Length": "8"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-804",
                                         "CandidateCUI": "C0427536",
                                         "CandidateMatched": "blood cells type white",
                                         "CandidatePreferred": "white blood cell type",
                                         "MatchedWords": ["blood","cells","type","white"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "7",
                                                 "TextMatchEnd": "8",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             },
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "3",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "1"
                                             },
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "4",
                                                 "ConcMatchEnd": "4",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "411",
                                                 "Length": "5"
                                             },
                                             {
                                                 "StartPos": "420",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-769",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-744",
                                         "CandidateCUI": "C1552740",
                                         "CandidateMatched": "specific",
                                         "CandidatePreferred": "Entity Determiner - specific",
                                         "MatchedWords": ["specific"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V3.0","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "402",
                                                 "Length": "8"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-804",
                                         "CandidateCUI": "C0427536",
                                         "CandidateMatched": "blood cells type white",
                                         "CandidatePreferred": "white blood cell type",
                                         "MatchedWords": ["blood","cells","type","white"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "7",
                                                 "TextMatchEnd": "8",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "2",
                                                 "LexVariation": "0"
                                             },
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "3",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "1"
                                             },
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "4",
                                                 "ConcMatchEnd": "4",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "411",
                                                 "Length": "5"
                                             },
                                             {
                                                 "StartPos": "420",
                                                 "Length": "17"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "called",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "verb",
                                 "LexMatch": "called",
                                 "InputMatch": "called",
                                 "LexCat": "verb",
                                 "Tokens": ["called"]
                             }],
                         "PhraseStartPos": "438",
                         "PhraseLength": "6",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-966",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-966",
                                         "CandidateCUI": "C0679006",
                                         "CandidateMatched": "Call",
                                         "CandidatePreferred": "Decision",
                                         "MatchedWords": ["call"],
                                         "SemTypes": ["menp"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "1"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "438",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-966",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-966",
                                         "CandidateCUI": "C1947967",
                                         "CandidateMatched": "Call",
                                         "CandidatePreferred": "Call (Instruction)",
                                         "MatchedWords": ["call"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "1"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "438",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "lymphocytes",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "lymphocytes",
                                 "InputMatch": "lymphocytes",
                                 "LexCat": "noun",
                                 "Tokens": ["lymphocytes"]
                             }],
                         "PhraseStartPos": "445",
                         "PhraseLength": "11",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0024264",
                                         "CandidateMatched": "Lymphocytes",
                                         "CandidatePreferred": "Lymphocyte",
                                         "MatchedWords": ["lymphocytes"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","HL7V2.5","LCH","LCH_NW","LNC","MSH","MTH","NCI","NCI_NCI-GLOSS","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "445",
                                                 "Length": "11"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C4018897",
                                         "CandidateMatched": "Lymphocytes",
                                         "CandidatePreferred": "Lymphocyte component of blood",
                                         "MatchedWords": ["lymphocytes"],
                                         "SemTypes": ["bdsu"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "445",
                                                 "Length": "11"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "or",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "conj",
                                 "LexMatch": "or",
                                 "InputMatch": "or",
                                 "LexCat": "conj",
                                 "Tokens": ["or"]
                             }],
                         "PhraseStartPos": "457",
                         "PhraseLength": "2",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "lymphoblasts.",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "lymphoblasts",
                                 "InputMatch": "lymphoblasts",
                                 "LexCat": "noun",
                                 "Tokens": ["lymphoblasts"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ".",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "460",
                         "PhraseLength": "13",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0229613",
                                         "CandidateMatched": "Lymphoblasts",
                                         "CandidatePreferred": "lymphoblast",
                                         "MatchedWords": ["lymphoblasts"],
                                         "SemTypes": ["cell"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","FMA","LNC","MTH","NCI","NCI_NCI-GLOSS","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "460",
                                                 "Length": "12"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C1167770",
                                         "CandidateMatched": "Lymphoblasts",
                                         "CandidatePreferred": "Lymphoblast count",
                                         "MatchedWords": ["lymphoblasts"],
                                         "SemTypes": ["lbpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI","NCI_CDISC"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "460",
                                                 "Length": "12"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     }]
             },
             {
                 "PMID": "00000000",
                 "UttSection": "tx",
                 "UttNum": "7",
                 "UttText": "ALL is the most common type of cancer in children.",
                 "UttStartPos": "474",
                 "UttLength": "50",
                 "Phrases": [
                     {
                         "PhraseText": "ALL",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "acute lymphocytic leukemia",
                                 "InputMatch": "acute lymphocytic leukemia",
                                 "LexCat": "noun",
                                 "Tokens": ["acute","lymphocytic","leukemia"]
                             }],
                         "PhraseStartPos": "474",
                         "PhraseLength": "3",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0023449",
                                         "CandidateMatched": "Acute Lymphocytic Leukaemia",
                                         "CandidatePreferred": "Acute lymphocytic leukemia",
                                         "MatchedWords": ["acute","lymphocytic","leukaemia"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","COSTAR","CSP","CST","HPO","ICD10CM","ICD9CM","MEDLINEPLUS","MTH","NCI","NCI_CDISC","NCI_CTEP-SDC","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","OMIM","PDQ","QMR","SNM","SNOMEDCT_US","SNOMEDCT_VET"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "474",
                                                 "Length": "3"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C1961102",
                                         "CandidateMatched": "Acute Lymphocytic Leukemia",
                                         "CandidatePreferred": "Precursor Cell Lymphoblastic Leukemia Lymphoma",
                                         "MatchedWords": ["acute","lymphocytic","leukemia"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "3",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["LCH_NW","MEDLINEPLUS","MSH","MTH","NDFRT","NLMSubSyn","OMIM","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "474",
                                                 "Length": "3"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "is",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "aux",
                                 "LexMatch": "is",
                                 "InputMatch": "is",
                                 "LexCat": "aux",
                                 "Tokens": ["is"]
                             }],
                         "PhraseStartPos": "478",
                         "PhraseLength": "2",
                         "Candidates": [],
                         "Mappings": []
                     },
                     {
                         "PhraseText": "the most common type of cancer",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "det",
                                 "LexMatch": "the",
                                 "InputMatch": "the",
                                 "LexCat": "det",
                                 "Tokens": ["the"]
                             },
                             {
                                 "SyntaxType": "adv",
                                 "LexMatch": "most",
                                 "InputMatch": "most",
                                 "LexCat": "adv",
                                 "Tokens": ["most"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "common",
                                 "InputMatch": "common",
                                 "LexCat": "adj",
                                 "Tokens": ["common"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "type",
                                 "InputMatch": "type",
                                 "LexCat": "noun",
                                 "Tokens": ["type"]
                             },
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "of",
                                 "InputMatch": "of",
                                 "LexCat": "prep",
                                 "Tokens": ["of"]
                             },
                             {
                                 "SyntaxType": "mod",
                                 "LexMatch": "cancer",
                                 "InputMatch": "cancer",
                                 "LexCat": "noun",
                                 "Tokens": ["cancer"]
                             }],
                         "PhraseStartPos": "481",
                         "PhraseLength": "30",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205214",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "Common (qualifier value)",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0006826",
                                         "CandidateMatched": "CANCER",
                                         "CandidatePreferred": "Malignant Neoplasms",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AIR","AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_FDA","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205214",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "Common (qualifier value)",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0998265",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Cancer Genus",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["euka"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCBI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205214",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "Common (qualifier value)",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1306459",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Primary malignant neoplasm",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205214",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "Common (qualifier value)",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0006826",
                                         "CandidateMatched": "CANCER",
                                         "CandidatePreferred": "Malignant Neoplasms",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AIR","AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_FDA","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205214",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "Common (qualifier value)",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0998265",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Cancer Genus",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["euka"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCBI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205214",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "Common (qualifier value)",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1306459",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Primary malignant neoplasm",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1522138",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "shared attribute",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0006826",
                                         "CandidateMatched": "CANCER",
                                         "CandidatePreferred": "Malignant Neoplasms",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AIR","AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_FDA","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1522138",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "shared attribute",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0998265",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Cancer Genus",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["euka"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCBI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1522138",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "shared attribute",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1306459",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Primary malignant neoplasm",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1522138",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "shared attribute",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0006826",
                                         "CandidateMatched": "CANCER",
                                         "CandidatePreferred": "Malignant Neoplasms",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AIR","AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_FDA","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1522138",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "shared attribute",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0998265",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Cancer Genus",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["euka"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCBI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1522138",
                                         "CandidateMatched": "Common",
                                         "CandidatePreferred": "shared attribute",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["ftcn"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["MTH","NCI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1306459",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Primary malignant neoplasm",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C3245511",
                                         "CandidateMatched": "common",
                                         "CandidatePreferred": "Common Specifications in HL7 V3 Publishing",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V3.0","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0006826",
                                         "CandidateMatched": "CANCER",
                                         "CandidatePreferred": "Malignant Neoplasms",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AIR","AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_FDA","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C3245511",
                                         "CandidateMatched": "common",
                                         "CandidatePreferred": "Common Specifications in HL7 V3 Publishing",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V3.0","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0998265",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Cancer Genus",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["euka"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCBI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C3245511",
                                         "CandidateMatched": "common",
                                         "CandidatePreferred": "Common Specifications in HL7 V3 Publishing",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V3.0","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C0332307",
                                         "CandidateMatched": "TYPE",
                                         "CandidatePreferred": "Type - attribute",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qlco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","NCI_CareLex","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1306459",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Primary malignant neoplasm",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C3245511",
                                         "CandidateMatched": "common",
                                         "CandidatePreferred": "Common Specifications in HL7 V3 Publishing",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V3.0","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0006826",
                                         "CandidateMatched": "CANCER",
                                         "CandidatePreferred": "Malignant Neoplasms",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["AIR","AOD","CCS","CCS_10","CHV","COSTAR","CSP","CST","DXP","HPO","ICD10CM","ICD9CM","LCH","LCH_NW","LNC","MEDLINEPLUS","MSH","MTH","MTHICD9","NCI","NCI_CDISC","NCI_FDA","NCI_NCI-GLOSS","NCI_NICHD","NLMSubSyn","PDQ","SNM","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C3245511",
                                         "CandidateMatched": "common",
                                         "CandidatePreferred": "Common Specifications in HL7 V3 Publishing",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V3.0","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0998265",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Cancer Genus",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["euka"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCBI"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-697",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C0205393",
                                         "CandidateMatched": "Most",
                                         "CandidatePreferred": "Most",
                                         "MatchedWords": ["most"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "2",
                                                 "TextMatchEnd": "2",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","LNC","MTH","NCI","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "485",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C3245511",
                                         "CandidateMatched": "common",
                                         "CandidatePreferred": "Common Specifications in HL7 V3 Publishing",
                                         "MatchedWords": ["common"],
                                         "SemTypes": ["inpr"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "3",
                                                 "TextMatchEnd": "3",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V3.0","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "490",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-753",
                                         "CandidateCUI": "C1547052",
                                         "CandidateMatched": "*Type",
                                         "CandidatePreferred": "*Type - Kind of quantity",
                                         "MatchedWords": ["type"],
                                         "SemTypes": ["qnco"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "4",
                                                 "TextMatchEnd": "4",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["HL7V2.5","MTH"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "497",
                                                 "Length": "4"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     },
                                     {
                                         "CandidateScore": "-586",
                                         "CandidateCUI": "C1306459",
                                         "CandidateMatched": "Cancer",
                                         "CandidatePreferred": "Primary malignant neoplasm",
                                         "MatchedWords": ["cancer"],
                                         "SemTypes": ["neop"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "6",
                                                 "TextMatchEnd": "6",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "no",
                                         "IsOverMatch": "no",
                                         "Sources": ["CHV","MTH","NCI","NLMSubSyn","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "505",
                                                 "Length": "6"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     },
                     {
                         "PhraseText": "in children.",
                         "SyntaxUnits": [
                             {
                                 "SyntaxType": "prep",
                                 "LexMatch": "in",
                                 "InputMatch": "in",
                                 "LexCat": "prep",
                                 "Tokens": ["in"]
                             },
                             {
                                 "SyntaxType": "head",
                                 "LexMatch": "children",
                                 "InputMatch": "children",
                                 "LexCat": "noun",
                                 "Tokens": ["children"]
                             },
                             {
                                 "SyntaxType": "punc",
                                 "InputMatch": ".",
                                 "Tokens": []
                             }],
                         "PhraseStartPos": "512",
                         "PhraseLength": "12",
                         "Candidates": [],
                         "Mappings": [
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0008059",
                                         "CandidateMatched": "Children",
                                         "CandidatePreferred": "Child",
                                         "MatchedWords": ["children"],
                                         "SemTypes": ["aggp"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","CSP","DXP","HL7V3.0","LCH","LCH_NW","LNC","MSH","MTH","NCI","NCI_FDA","NCI_NICHD","NDFRT","SNMI","SNOMEDCT_US"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "515",
                                                 "Length": "8"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             },
                             {
                                 "MappingScore": "-1000",
                                 "MappingCandidates": [
                                     {
                                         "CandidateScore": "-1000",
                                         "CandidateCUI": "C0680063",
                                         "CandidateMatched": "Children",
                                         "CandidatePreferred": "Offspring",
                                         "MatchedWords": ["children"],
                                         "SemTypes": ["famg"],
                                         "MatchMaps": [
                                             {
                                                 "TextMatchStart": "1",
                                                 "TextMatchEnd": "1",
                                                 "ConcMatchStart": "1",
                                                 "ConcMatchEnd": "1",
                                                 "LexVariation": "0"
                                             }],
                                         "IsHead": "yes",
                                         "IsOverMatch": "no",
                                         "Sources": ["AOD","CHV","HL7V3.0","MTH","NCI","NCI_CDISC","NCI_NCI-GLOSS"],
                                         "ConceptPIs": [
                                             {
                                                 "StartPos": "515",
                                                 "Length": "8"
                                             }],
                                         "Status": "0",
                                         "Negated": "0"
                                     }]
                             }]
                     }]
             }]
     }
 }
]}
