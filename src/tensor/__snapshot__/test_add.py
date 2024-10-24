import torch

def test_add():
    a = torch.tensor([[[[0.23193953931331635,-0.9572423696517944,-0.6499392986297607,-0.9692349433898926,-0.3362119793891907],[-0.4338535666465759,-0.8236226439476013,0.6791391968727112,0.6082472801208496,0.3328486680984497],[0.05109986662864685,0.7491360306739807,-0.6769163012504578,-0.01848955824971199,0.3614869713783264],[-0.6127452254295349,-0.26850271224975586,-0.1907285451889038,0.3077051341533661,-0.34001031517982483]],[[0.3663826584815979,-0.14738808572292328,0.3900488317012787,-0.06342953443527222,0.47320351004600525],[-0.15762098133563995,-0.03688250482082367,0.3375449478626251,-0.3398120701313019,-0.6835131049156189],[0.7505687475204468,-0.023452138528227806,0.024893853813409805,0.4261877238750458,0.13734762370586395],[0.2711954414844513,-0.32425403594970703,-0.02062375470995903,-0.6134109497070312,0.6571998596191406]],[[-0.36438438296318054,-0.810602068901062,0.8760115504264832,0.33559539914131165,0.5282369256019592],[0.45072439312934875,-0.7516795992851257,0.7518489360809326,0.676425039768219,0.34842780232429504],[-0.90400230884552,-0.9477491974830627,-0.9462980031967163,-0.4534570574760437,-0.3560742139816284],[-0.6082800626754761,-0.4886247515678406,-0.09143755584955215,0.9139198064804077,0.2639783024787903]]],[[[-0.45535042881965637,-0.5269649028778076,-0.22615283727645874,-0.9469784498214722,0.1851653903722763],[0.019958922639489174,0.33514103293418884,-0.5552229881286621,0.0343792624771595,-0.695143461227417],[0.10260666906833649,0.29081040620803833,-0.8598014712333679,-0.8798351287841797,-0.42887255549430847],[-0.4169297516345978,0.6753455996513367,-0.417716920375824,0.94532710313797,0.41667166352272034]],[[-0.1105073019862175,0.4210078716278076,-0.45518267154693604,-0.34820571541786194,-0.7651030421257019],[-0.9649389386177063,-0.8886757493019104,-0.13860481977462769,-0.768852949142456,0.3825380802154541],[0.6009319424629211,0.8871250748634338,-0.5087488889694214,-0.5200148224830627,-0.05050414055585861],[-0.018569685518741608,-0.30058401823043823,-0.5005064606666565,0.3609837293624878,-0.6083757281303406]],[[0.4932307302951813,-0.9541529417037964,0.58220374584198,0.35814064741134644,0.24016596376895905],[0.556648313999176,-0.14718855917453766,-0.37906908988952637,-0.22143657505512238,0.40061482787132263],[0.6845512986183167,0.5608916282653809,-0.8602480292320251,0.3451920449733734,-0.2965222895145416],[0.30276188254356384,0.020035596564412117,-0.6529751420021057,0.29793062806129456,0.7429209351539612]]]])
    b = torch.tensor([[[[0.034394923597574234,0.48843833804130554,0.9946474432945251,0.8388859033584595,-0.8829470872879028],[0.8432904481887817,0.9531295895576477,0.9591870307922363,0.15551701188087463,-0.6000486016273499],[-0.13492633402347565,-0.5275817513465881,-0.9949140548706055,0.7311679124832153,-0.4301653206348419],[-0.12583136558532715,0.7397727370262146,0.17758455872535706,-0.1129089817404747,0.3744562268257141]],[[0.7054449915885925,-0.4413159489631653,0.8792769312858582,0.7953783273696899,0.08021970838308334],[-0.7728646993637085,-0.007905203849077225,-0.7004809975624084,0.5250273942947388,-0.13928593695163727],[0.21956875920295715,-0.6038710474967957,0.5634909272193909,-0.6904449462890625,0.26651284098625183],[0.9239564538002014,0.7150423526763916,0.27370014786720276,-0.9700709581375122,-0.19397872686386108]],[[-0.018409384414553642,-0.29282984137535095,0.01251191645860672,0.19382846355438232,0.5944182276725769],[0.5965877175331116,-0.8085740208625793,0.061679303646087646,-0.6684167981147766,-0.6203181147575378],[-0.4517829120159149,0.06547470390796661,0.48156294226646423,0.6527250409126282,0.07879525423049927],[-0.07892513275146484,-0.710239589214325,-0.23844951391220093,0.9620501399040222,0.13390153646469116]]],[[[0.31101807951927185,0.5514454245567322,-0.5808465480804443,-0.6126863956451416,0.8685460090637207],[-0.4559074640274048,0.7222825288772583,0.8360690474510193,0.2554662823677063,0.501268744468689],[-0.759762704372406,-0.5554875135421753,-0.8661314249038696,-0.6899222135543823,0.07177434116601944],[-0.6469445824623108,-0.22397305071353912,-0.8917180299758911,0.04906691983342171,-0.6261913180351257]],[[0.8172567486763,-0.007235459517687559,-0.18060649931430817,0.05156683549284935,-0.23014697432518005],[-0.6069337725639343,-0.5442634224891663,0.40128758549690247,0.7460659742355347,-0.6252037286758423],[0.5430262684822083,-0.47781214118003845,0.7129918336868286,-0.5880556702613831,-0.8625831604003906],[0.08878526091575623,0.7582480311393738,-0.4888702929019928,0.2053065299987793,-0.037133850157260895]],[[-0.0012097560102120042,-0.9575442671775818,-0.6907637715339661,-0.36052557826042175,-0.6071164011955261],[-0.14524538815021515,-0.06849785894155502,0.2890911400318146,0.47452911734580994,-0.018064215779304504],[-0.12388404458761215,0.44752970337867737,-0.9238727688789368,-0.5709429979324341,-0.8505104780197144],[0.012452846392989159,-0.23402439057826996,-0.9179116487503052,0.8087320923805237,0.14766158163547516]]]])
    c = a + b
    assert (c == torch.tensor([[[[0.2663344740867615,-0.4688040316104889,0.3447081446647644,-0.1303490400314331,-1.2191591262817383],[0.4094368815422058,0.1295069456100464,1.6383261680603027,0.7637642621994019,-0.26719993352890015],[-0.0838264673948288,0.22155427932739258,-1.671830415725708,0.7126783728599548,-0.0686783492565155],[-0.7385765910148621,0.47127002477645874,-0.013143986463546753,0.1947961449623108,0.03444591164588928]],[[1.0718276500701904,-0.5887040495872498,1.2693257331848145,0.7319487929344177,0.5534232258796692],[-0.9304856657981873,-0.044787708669900894,-0.3629360496997833,0.1852153241634369,-0.822799026966095],[0.9701374769210815,-0.6273232102394104,0.5883848071098328,-0.2642572224140167,0.4038604497909546],[1.195151925086975,0.39078831672668457,0.2530764043331146,-1.5834819078445435,0.46322113275527954]],[[-0.38279375433921814,-1.1034319400787354,0.8885234594345093,0.5294238328933716,1.1226551532745361],[1.0473121404647827,-1.560253620147705,0.8135282397270203,0.008008241653442383,-0.2718903124332428],[-1.3557852506637573,-0.8822745084762573,-0.4647350609302521,0.19926798343658447,-0.27727895975112915],[-0.6872051954269409,-1.1988643407821655,-0.3298870623111725,1.8759698867797852,0.39787983894348145]]],[[[-0.14433234930038452,0.02448052167892456,-0.8069993853569031,-1.5596648454666138,1.0537114143371582],[-0.43594855070114136,1.0574235916137695,0.2808460593223572,0.2898455560207367,-0.19387471675872803],[-0.6571560502052307,-0.26467710733413696,-1.7259328365325928,-1.569757342338562,-0.35709822177886963],[-1.063874363899231,0.45137256383895874,-1.3094348907470703,0.9943940043449402,-0.2095196545124054]],[[0.706749439239502,0.4137724041938782,-0.635789155960083,-0.2966388761997223,-0.9952499866485596],[-1.5718727111816406,-1.4329391717910767,0.2626827657222748,-0.022786974906921387,-0.24266564846038818],[1.1439582109451294,0.4093129336833954,0.20424294471740723,-1.1080704927444458,-0.9130873084068298],[0.07021557539701462,0.45766401290893555,-0.9893767833709717,0.5662902593612671,-0.6455096006393433]],[[0.49202096462249756,-1.9116971492767334,-0.10856002569198608,-0.0023849308490753174,-0.36695045232772827],[0.41140294075012207,-0.21568641066551208,-0.08997794985771179,0.25309252738952637,0.38255059719085693],[0.5606672763824463,1.0084213018417358,-1.784120798110962,-0.22575095295906067,-1.1470327377319336],[0.31521472334861755,-0.213988795876503,-1.5708868503570557,1.1066627502441406,0.8905825018882751]]]])).all()