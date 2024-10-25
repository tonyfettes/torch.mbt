import torch

def test_linear(snapshot):
    input = torch.tensor([[-0.4301600456237793,0.7356255650520325,0.7850341796875,0.32139524817466736,0.6747738718986511,0.8503944277763367,0.2769254148006439,-0.5336059927940369]])
    weight = torch.tensor([[-0.10449987649917603,-0.2891155183315277,-0.3181218206882477,-0.054641243070364,-0.19516339898109436,-0.03352576121687889,0.23366543650627136,0.07774727046489716,0.15250012278556824,0.3387894034385681,0.25557899475097656,-0.18343213200569153,0.09550005942583084,-0.07738534361124039,0.3337253928184509,-0.1593637615442276],[-0.19356603920459747,0.14759302139282227,0.13963159918785095,0.3526599109172821,-0.31833621859550476,0.06593333929777145,-0.25350186228752136,-0.09950438886880875,-0.008569345809519291,0.08425828069448471,0.17673572897911072,-0.08864541351795197,0.165083110332489,-0.09248556941747665,-0.15438318252563477,-0.054068561643362045],[-0.2785130739212036,0.17523883283138275,-0.19689783453941345,0.18168550729751587,0.2974433898925781,-0.19166414439678192,-0.31767889857292175,0.3115870952606201,0.23257069289684296,0.16344891488552094,-0.1908106505870819,-0.1677696406841278,0.28765973448753357,0.1419617384672165,-0.3493776023387909,0.2712598145008087],[0.10740985721349716,0.16186952590942383,-0.1090780720114708,-0.0881650447845459,0.20831342041492462,-0.267898827791214,0.24084877967834473,-0.34765011072158813,-0.2884160280227661,0.26205429434776306,0.23972302675247192,-0.2877962589263916,-0.2823966443538666,-0.15185387432575226,0.3270179331302643,0.2118854522705078],[0.19171930849552155,-0.05846284329891205,-0.18643680214881897,-0.023307643830776215,-0.03570830449461937,-0.07345201820135117,-0.11341727524995804,-0.04206091910600662,-0.3481800854206085,-0.13712257146835327,0.23458626866340637,0.3158166706562042,-0.2908225953578949,-0.05178840085864067,0.325983464717865,0.13402049243450165],[0.058022305369377136,-0.01930900663137436,-0.1159118190407753,-0.09195136278867722,0.18518081307411194,0.05592884495854378,0.27743178606033325,0.061755601316690445,-0.3469058871269226,0.297254353761673,-0.25413668155670166,-0.3055642247200012,0.3320108950138092,0.3450635075569153,-0.25547969341278076,-0.12612955272197723],[0.17797116935253143,0.2091580480337143,0.059001512825489044,-0.09701649099588394,0.010744853876531124,0.22199706733226776,0.14104336500167847,-0.2397661805152893,0.11481110751628876,0.33735227584838867,-0.17626847326755524,0.3058389127254486,-0.1337548941373825,0.1429482102394104,0.1424047350883484,0.3110893666744232],[0.17239980399608612,0.1575177013874054,0.2148265391588211,0.3434791564941406,0.10334490984678268,-0.2314218282699585,-0.066314697265625,-0.03836636245250702,-0.2295268476009369,0.05670378729701042,-0.3268022835254669,0.3251144289970398,-0.0476735457777977,-0.02064116857945919,-0.011810533702373505,0.28672173619270325]])
    bias = torch.tensor([-0.2995610535144806,0.25378626585006714,-0.0716765969991684,-0.0933355763554573,-0.16303308308124542,-0.21767345070838928,0.16216640174388885,0.2216375470161438,0.1585126519203186,0.1288575679063797,-0.33302974700927734,0.08836913853883743,0.3518984913825989,-0.21406416594982147,0.21930983662605286,0.24214544892311096])
    output = torch.nn.functional.linear(input, weight.transpose(-2, -1), bias)
    assert output - torch.tensor([[-0.4451225996017456,0.5943173170089722,-0.344412624835968,-0.00018315762281417847,0.06840808689594269,-0.2083512842655182,-0.06297081708908081,0.22607941925525665,-0.19918888807296753,0.4810630679130554,-0.31795865297317505,-0.25767049193382263,0.6418188214302063,0.12292371690273285,-0.1585407555103302,0.4682961106300354]]) == snapshot
