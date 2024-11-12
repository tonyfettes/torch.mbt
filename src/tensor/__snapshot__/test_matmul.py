import torch
def test_matmul_1x2_2x4():
    a = torch.tensor([[-302.6639404296875,-837.3680419921875]], dtype=torch.float32)
    b = torch.tensor([[-921.3792114257812,-158.2579345703125,-565.2535400390625,-97.1009521484375],[676.7674560546875,225.18017578125,441.6875,981.2388916015625]], dtype=torch.float32)
    c = torch.matmul(a, b)
    assert torch.allclose(c, torch.tensor([[-287835.1875,-140659.71875,-198773.140625,-792269.125]], dtype=torch.float32))
def test_matmul_1x8_8x16():
    a = torch.tensor([[740.23583984375,-531.2762451171875,276.59765625,-224.13189697265625,966.572021484375,-461.5667724609375,-560.627197265625,427.47509765625]], dtype=torch.float32)
    b = torch.tensor([[404.416259765625,1021.4122314453125,-922.0001831054688,190.96337890625,-734.2198486328125,-288.19549560546875,-24.8194580078125,244.0380859375,511.88134765625,-256.74456787109375,478.1317138671875,-267.8668212890625,-447.1414794921875,-156.59930419921875,-806.6600341796875,507.5458984375],[-570.277099609375,526.2174072265625,861.488037109375,-555.11865234375,-920.0963745117188,902.45263671875,673.5966796875,473.398681640625,-552.6466674804688,-485.91278076171875,833.151611328125,411.1650390625,-1011.9056396484375,785.65234375,311.0921630859375,468.82421875],[-315.923828125,-255.353271484375,603.340087890625,-775.9179077148438,697.572509765625,-1006.9022827148438,-835.342041015625,758.9903564453125,694.31201171875,-833.5469970703125,-817.9080200195312,-439.8157958984375,947.1451416015625,613.685791015625,555.2784423828125,-169.32647705078125],[-539.9786376953125,-67.50616455078125,-103.42230224609375,-212.73980712890625,-328.49151611328125,-121.8214111328125,-1008.437255859375,-397.14935302734375,679.4344482421875,914.7027587890625,-842.3121337890625,-149.9952392578125,944.14892578125,388.164794921875,168.050537109375,-55.92486572265625],[-335.71649169921875,-266.319580078125,536.3409423828125,161.9871826171875,803.5283203125,178.86328125,-1004.7467041015625,860.9405517578125,-736.0584716796875,-885.008544921875,961.606201171875,999.41064453125,-739.9482421875,-365.31024169921875,515.4595947265625,605.786376953125],[170.8865966796875,-280.98974609375,31.120361328125,642.97216796875,408.5052490234375,-694.43701171875,332.5284423828125,977.0765380859375,-510.52801513671875,885.80419921875,-387.39556884765625,414.022216796875,412.4481201171875,901.0111083984375,499.3231201171875,456.219970703125],[622.2041015625,994.8218994140625,299.31884765625,-670.2691650390625,-192.06787109375,-111.120849609375,-664.78076171875,164.231689453125,-946.5205078125,941.6319580078125,-138.07733154296875,-59.783203125,-34.20697021484375,830.434814453125,-867.6214599609375,735.0435791015625],[-207.59759521484375,-270.32867431640625,-472.19427490234375,-630.4496459960938,469.68408203125,641.930908203125,459.1016845703125,373.21142578125,-964.5571899414062,255.9443359375,1019.2069091796875,-619.9960327148438,635.1890869140625,701.328125,-440.48388671875,753.280517578125]], dtype=torch.float32)
    c = torch.matmul(a, b)
    assert torch.allclose(c, torch.tensor([[-204956.3125,-379985.21875,-815732.625,235411.25,1108474.125,-113866.90625,-936964.5,676729.875,354793.25,-2050245.625,1495220.25,38626.53125,-357905.21875,-1385314.5,-80597.234375,377211.125]], dtype=torch.float32))
def test_matmul_multi_2x3_3x2_2x4_4x2():
    a = torch.tensor([[803.875,329.1087646484375,690.968505859375],[870.803955078125,283.5716552734375,-546.4124755859375]], dtype=torch.float32)
    b = torch.tensor([[-421.4267578125,-1005.5994873046875],[924.0240478515625,-826.9221801757812],[-518.3827514648438,650.4935302734375]], dtype=torch.float32)
    c = torch.matmul(a, b)
    d = torch.tensor([[120.0782470703125,-162.73797607421875,398.11572265625,-739.593017578125],[683.60107421875,-916.1008911132812,-428.4652099609375,892.9178466796875]], dtype=torch.float32)
    e = torch.tensor([[-728.7808837890625,-446.03887939453125],[-900.9910888671875,354.2513427734375],[746.416015625,-557.7615966796875],[-264.1495361328125,-274.38201904296875]], dtype=torch.float32)
    f = torch.matmul(d, e)
    g = torch.matmul(c, f)
    assert torch.allclose(g, torch.tensor([[-72535261184,452212031488],[433209081856,908100894720]], dtype=torch.float32))
