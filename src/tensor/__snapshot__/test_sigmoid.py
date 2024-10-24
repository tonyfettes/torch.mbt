import torch

def test_sigmoid():
    a = torch.tensor([[[[-0.2955702841281891,-0.8177422285079956,-0.8997843861579895,-0.154548779129982,-0.5520054697990417],[-0.09482517838478088,0.660905659198761,0.21990250051021576,0.4313355088233948,0.9582411050796509],[0.7228865623474121,-0.5188244581222534,0.2701149582862854,-0.21887879073619843,0.9439179301261902],[-0.4507488012313843,-0.5474874377250671,0.41745609045028687,0.39493778347969055,0.997472882270813]],[[-0.9003908038139343,0.18648764491081238,-0.717011570930481,-0.2814409136772156,-0.024237770587205887],[0.23831841349601746,0.49988412857055664,-0.2507270872592926,0.4669255316257477,-0.2615886926651001],[-0.43666160106658936,-0.15292899310588837,-0.7877539396286011,0.4956502616405487,-0.5569111704826355],[0.5138841867446899,0.841296911239624,-0.5421080589294434,-0.8985316157341003,0.8813014030456543]],[[0.6578092575073242,0.4623033404350281,-0.5396940112113953,-0.47452419996261597,0.8136246204376221],[0.40152841806411743,-0.9881891012191772,0.7672386169433594,0.3038009703159332,0.4578361511230469],[-0.3085193634033203,-0.24936839938163757,0.5891993641853333,-0.7577323317527771,0.6812232136726379],[-0.9833030104637146,-0.8157637119293213,0.7412014603614807,0.6780391335487366,-0.814010739326477]]],[[[-0.7987383008003235,-0.42950761318206787,0.9249464273452759,0.5993025898933411,0.5422641038894653],[-0.16535788774490356,-0.5273228883743286,-0.06592397391796112,-0.10099833458662033,-0.20775368809700012],[-0.3207924962043762,-0.1189662367105484,-0.9848020076751709,-0.38784119486808777,0.6635101437568665],[0.8932644128799438,-0.8225705027580261,-0.14647971093654633,0.9220204949378967,0.37906718254089355]],[[0.16411185264587402,-0.0546141155064106,-0.3278481364250183,-0.26007774472236633,0.5237704515457153],[0.15819066762924194,0.784695565700531,0.1746712177991867,-0.9811980128288269,0.8407623171806335],[-0.7188071012496948,-0.8642661571502686,0.939068615436554,0.9759870171546936,-0.7226056456565857],[-0.3567482829093933,0.5033784508705139,0.5915883183479309,0.1668814718723297,-0.2744040787220001]],[[0.030391035601496696,0.6279025673866272,0.3989308774471283,-0.6781611442565918,0.3247348666191101],[0.9541763067245483,-0.4985625147819519,0.8650431036949158,-0.3783159852027893,0.40431860089302063],[0.4027813971042633,0.8798936605453491,0.4876202642917633,0.4455273449420929,0.6076212525367737],[0.9715057611465454,0.29230356216430664,-0.6545597314834595,-0.18756628036499023,-0.10851645469665527]]]])
    c = torch.sigmoid(a)
    assert torch.allclose(c, torch.tensor([[[[0.42664071917533875,0.3062431216239929,0.28909480571746826,0.4614395201206207,0.36539924144744873],[0.47631141543388367,0.6594638228416443,0.5547551512718201,0.6061925292015076,0.7227694988250732],[0.6732423901557922,0.37312716245651245,0.567121148109436,0.4454977214336395,0.7198904156684875],[0.3891827166080475,0.3664475381374359,0.6028743386268616,0.597470760345459,0.7305614352226257]],[[0.2889701724052429,0.5464872717857361,0.3280513882637024,0.43010056018829346,0.49394088983535767],[0.5592992305755615,0.6224320530891418,0.4376445412635803,0.6146557927131653,0.43497321009635925],[0.3925367295742035,0.4618420898914337,0.3126511573791504,0.6214365363121033,0.3642624616622925],[0.6257165670394897,0.6987382769584656,0.3676973283290863,0.2893523573875427,0.7070918083190918]],[[0.6587681174278259,0.6135604381561279,0.3682587742805481,0.38354599475860596,0.6928813457489014],[0.5990548133850098,0.27126991748809814,0.6829232573509216,0.5753714442253113,0.6125007271766663],[0.4234762191772461,0.43797898292541504,0.6431813836097717,0.3191387951374054,0.6640116572380066],[0.2722368836402893,0.3066636323928833,0.6772584915161133,0.6633009314537048,0.30703648924827576]]],[[[0.3102954924106598,0.39424389600753784,0.7160488963127136,0.6454967260360718,0.6323389410972595],[0.4587544798851013,0.3711414933204651,0.48352497816085815,0.4747718572616577,0.44824761152267456],[0.4204826056957245,0.47029349207878113,0.2719399929046631,0.4042370617389679,0.6600484848022461],[0.7095633745193481,0.30521827936172485,0.46344539523124695,0.7154536247253418,0.5936480760574341]],[[0.5409361124038696,0.48634985089302063,0.4187643229961395,0.4353446066379547,0.6280289888381958],[0.5394653677940369,0.6866912245750427,0.5435571074485779,0.27265414595603943,0.6986257433891296],[0.32765573263168335,0.29644879698753357,0.7189114689826965,0.7263112664222717,0.3268194794654846],[0.41174694895744324,0.6232529282569885,0.6437295079231262,0.5416238307952881,0.43182623386383057]],[[0.5075972080230713,0.6520137190818787,0.5984307527542114,0.3366718292236328,0.5804777145385742],[0.7219542860984802,0.3778785467147827,0.7037132382392883,0.4065331220626831,0.5997248291969299],[0.5993557572364807,0.7068002223968506,0.6195456385612488,0.6095752716064453,0.6473979949951172],[0.7254195213317871,0.5725600123405457,0.3419627845287323,0.4532454013824463,0.472897469997406]]]]))
