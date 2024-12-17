import torch

_layer0 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer0.weight = torch.nn.Parameter(torch.tensor([[-0.2955702543258667]]).transpose(-2, -1))
    _layer0.bias = torch.nn.Parameter(torch.tensor([-0.8177422285079956]))
_layer1 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer1.weight = torch.nn.Parameter(torch.tensor([[-0.8997843861579895]]).transpose(-2, -1))
    _layer1.bias = torch.nn.Parameter(torch.tensor([-0.1545487642288208]))
tuple2 = torch.nn.Sequential(
    _layer0,
    _layer1,
)
_layer2 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer2.weight = torch.nn.Parameter(torch.tensor([[-0.552005410194397]]).transpose(-2, -1))
    _layer2.bias = torch.nn.Parameter(torch.tensor([-0.0948251485824585]))
_layer3 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer3.weight = torch.nn.Parameter(torch.tensor([[0.6609057188034058]]).transpose(-2, -1))
    _layer3.bias = torch.nn.Parameter(torch.tensor([0.21990251541137695]))
_layer4 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer4.weight = torch.nn.Parameter(torch.tensor([[0.43133544921875]]).transpose(-2, -1))
    _layer4.bias = torch.nn.Parameter(torch.tensor([0.9582411050796509]))
tuple3 = torch.nn.Sequential(
    _layer2,
    _layer3,
    _layer4,
)
_layer5 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer5.weight = torch.nn.Parameter(torch.tensor([[0.7228865623474121]]).transpose(-2, -1))
    _layer5.bias = torch.nn.Parameter(torch.tensor([-0.5188244581222534]))
_layer6 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer6.weight = torch.nn.Parameter(torch.tensor([[0.2701148986816406]]).transpose(-2, -1))
    _layer6.bias = torch.nn.Parameter(torch.tensor([-0.21887880563735962]))
_layer7 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer7.weight = torch.nn.Parameter(torch.tensor([[0.943917989730835]]).transpose(-2, -1))
    _layer7.bias = torch.nn.Parameter(torch.tensor([-0.4507488012313843]))
_layer8 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer8.weight = torch.nn.Parameter(torch.tensor([[-0.5474874973297119]]).transpose(-2, -1))
    _layer8.bias = torch.nn.Parameter(torch.tensor([0.41745615005493164]))
tuple4 = torch.nn.Sequential(
    _layer5,
    _layer6,
    _layer7,
    _layer8,
)
_layer9 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer9.weight = torch.nn.Parameter(torch.tensor([[0.39493775367736816]]).transpose(-2, -1))
    _layer9.bias = torch.nn.Parameter(torch.tensor([0.997472882270813]))
_layer10 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer10.weight = torch.nn.Parameter(torch.tensor([[-0.9003908038139343]]).transpose(-2, -1))
    _layer10.bias = torch.nn.Parameter(torch.tensor([0.18648767471313477]))
_layer11 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer11.weight = torch.nn.Parameter(torch.tensor([[-0.717011570930481]]).transpose(-2, -1))
    _layer11.bias = torch.nn.Parameter(torch.tensor([-0.2814409136772156]))
_layer12 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer12.weight = torch.nn.Parameter(torch.tensor([[-0.024237751960754395]]).transpose(-2, -1))
    _layer12.bias = torch.nn.Parameter(torch.tensor([0.23831844329833984]))
_layer13 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer13.weight = torch.nn.Parameter(torch.tensor([[0.49988412857055664]]).transpose(-2, -1))
    _layer13.bias = torch.nn.Parameter(torch.tensor([-0.250727117061615]))
tuple5 = torch.nn.Sequential(
    _layer9,
    _layer10,
    _layer11,
    _layer12,
    _layer13,
)
_layer14 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer14.weight = torch.nn.Parameter(torch.tensor([[0.4669255018234253]]).transpose(-2, -1))
    _layer14.bias = torch.nn.Parameter(torch.tensor([-0.2615886926651001]))
_layer15 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer15.weight = torch.nn.Parameter(torch.tensor([[-0.43666160106658936]]).transpose(-2, -1))
    _layer15.bias = torch.nn.Parameter(torch.tensor([-0.15292900800704956]))
_layer16 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer16.weight = torch.nn.Parameter(torch.tensor([[-0.7877539396286011]]).transpose(-2, -1))
    _layer16.bias = torch.nn.Parameter(torch.tensor([0.4956502914428711]))
_layer17 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer17.weight = torch.nn.Parameter(torch.tensor([[-0.5569112300872803]]).transpose(-2, -1))
    _layer17.bias = torch.nn.Parameter(torch.tensor([0.5138841867446899]))
_layer18 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer18.weight = torch.nn.Parameter(torch.tensor([[0.841296911239624]]).transpose(-2, -1))
    _layer18.bias = torch.nn.Parameter(torch.tensor([-0.5421080589294434]))
_layer19 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer19.weight = torch.nn.Parameter(torch.tensor([[-0.8985316157341003]]).transpose(-2, -1))
    _layer19.bias = torch.nn.Parameter(torch.tensor([0.8813014030456543]))
tuple6 = torch.nn.Sequential(
    _layer14,
    _layer15,
    _layer16,
    _layer17,
    _layer18,
    _layer19,
)
_layer20 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer20.weight = torch.nn.Parameter(torch.tensor([[0.6578092575073242]]).transpose(-2, -1))
    _layer20.bias = torch.nn.Parameter(torch.tensor([0.46230340003967285]))
_layer21 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer21.weight = torch.nn.Parameter(torch.tensor([[-0.5396940112113953]]).transpose(-2, -1))
    _layer21.bias = torch.nn.Parameter(torch.tensor([-0.47452419996261597]))
_layer22 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer22.weight = torch.nn.Parameter(torch.tensor([[0.8136246204376221]]).transpose(-2, -1))
    _layer22.bias = torch.nn.Parameter(torch.tensor([0.40152835845947266]))
_layer23 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer23.weight = torch.nn.Parameter(torch.tensor([[-0.9881891012191772]]).transpose(-2, -1))
    _layer23.bias = torch.nn.Parameter(torch.tensor([0.7672386169433594]))
_layer24 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer24.weight = torch.nn.Parameter(torch.tensor([[0.30380094051361084]]).transpose(-2, -1))
    _layer24.bias = torch.nn.Parameter(torch.tensor([0.4578361511230469]))
_layer25 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer25.weight = torch.nn.Parameter(torch.tensor([[-0.3085193634033203]]).transpose(-2, -1))
    _layer25.bias = torch.nn.Parameter(torch.tensor([-0.24936842918395996]))
_layer26 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer26.weight = torch.nn.Parameter(torch.tensor([[0.5891993045806885]]).transpose(-2, -1))
    _layer26.bias = torch.nn.Parameter(torch.tensor([-0.7577323317527771]))
tuple7 = torch.nn.Sequential(
    _layer20,
    _layer21,
    _layer22,
    _layer23,
    _layer24,
    _layer25,
    _layer26,
)
_layer27 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer27.weight = torch.nn.Parameter(torch.tensor([[0.6812231540679932]]).transpose(-2, -1))
    _layer27.bias = torch.nn.Parameter(torch.tensor([-0.9833030104637146]))
_layer28 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer28.weight = torch.nn.Parameter(torch.tensor([[-0.8157637119293213]]).transpose(-2, -1))
    _layer28.bias = torch.nn.Parameter(torch.tensor([0.7412015199661255]))
_layer29 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer29.weight = torch.nn.Parameter(torch.tensor([[0.6780390739440918]]).transpose(-2, -1))
    _layer29.bias = torch.nn.Parameter(torch.tensor([-0.814010739326477]))
_layer30 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer30.weight = torch.nn.Parameter(torch.tensor([[-0.7987383008003235]]).transpose(-2, -1))
    _layer30.bias = torch.nn.Parameter(torch.tensor([-0.42950761318206787]))
_layer31 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer31.weight = torch.nn.Parameter(torch.tensor([[0.9249464273452759]]).transpose(-2, -1))
    _layer31.bias = torch.nn.Parameter(torch.tensor([0.5993025302886963]))
_layer32 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer32.weight = torch.nn.Parameter(torch.tensor([[0.5422641038894653]]).transpose(-2, -1))
    _layer32.bias = torch.nn.Parameter(torch.tensor([-0.16535788774490356]))
_layer33 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer33.weight = torch.nn.Parameter(torch.tensor([[-0.5273228883743286]]).transpose(-2, -1))
    _layer33.bias = torch.nn.Parameter(torch.tensor([-0.06592398881912231]))
_layer34 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer34.weight = torch.nn.Parameter(torch.tensor([[-0.10099834203720093]]).transpose(-2, -1))
    _layer34.bias = torch.nn.Parameter(torch.tensor([-0.2077537178993225]))
tuple8 = torch.nn.Sequential(
    _layer27,
    _layer28,
    _layer29,
    _layer30,
    _layer31,
    _layer32,
    _layer33,
    _layer34,
)
_layer35 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer35.weight = torch.nn.Parameter(torch.tensor([[-0.3207924962043762]]).transpose(-2, -1))
    _layer35.bias = torch.nn.Parameter(torch.tensor([-0.11896622180938721]))
_layer36 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer36.weight = torch.nn.Parameter(torch.tensor([[-0.9848020076751709]]).transpose(-2, -1))
    _layer36.bias = torch.nn.Parameter(torch.tensor([-0.3878411650657654]))
_layer37 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer37.weight = torch.nn.Parameter(torch.tensor([[0.6635102033615112]]).transpose(-2, -1))
    _layer37.bias = torch.nn.Parameter(torch.tensor([0.8932644128799438]))
_layer38 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer38.weight = torch.nn.Parameter(torch.tensor([[-0.8225704431533813]]).transpose(-2, -1))
    _layer38.bias = torch.nn.Parameter(torch.tensor([-0.14647972583770752]))
_layer39 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer39.weight = torch.nn.Parameter(torch.tensor([[0.922020435333252]]).transpose(-2, -1))
    _layer39.bias = torch.nn.Parameter(torch.tensor([0.37906718254089355]))
_layer40 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer40.weight = torch.nn.Parameter(torch.tensor([[0.16411185264587402]]).transpose(-2, -1))
    _layer40.bias = torch.nn.Parameter(torch.tensor([-0.054614126682281494]))
_layer41 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer41.weight = torch.nn.Parameter(torch.tensor([[-0.3278481364250183]]).transpose(-2, -1))
    _layer41.bias = torch.nn.Parameter(torch.tensor([-0.26007771492004395]))
_layer42 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer42.weight = torch.nn.Parameter(torch.tensor([[0.5237704515457153]]).transpose(-2, -1))
    _layer42.bias = torch.nn.Parameter(torch.tensor([0.15819060802459717]))
_layer43 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer43.weight = torch.nn.Parameter(torch.tensor([[0.7846956253051758]]).transpose(-2, -1))
    _layer43.bias = torch.nn.Parameter(torch.tensor([0.17467117309570312]))
tuple9 = torch.nn.Sequential(
    _layer35,
    _layer36,
    _layer37,
    _layer38,
    _layer39,
    _layer40,
    _layer41,
    _layer42,
    _layer43,
)
_layer44 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer44.weight = torch.nn.Parameter(torch.tensor([[-0.9811979532241821]]).transpose(-2, -1))
    _layer44.bias = torch.nn.Parameter(torch.tensor([0.8407622575759888]))
_layer45 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer45.weight = torch.nn.Parameter(torch.tensor([[-0.7188071012496948]]).transpose(-2, -1))
    _layer45.bias = torch.nn.Parameter(torch.tensor([-0.8642661571502686]))
_layer46 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer46.weight = torch.nn.Parameter(torch.tensor([[0.9390685558319092]]).transpose(-2, -1))
    _layer46.bias = torch.nn.Parameter(torch.tensor([0.9759869575500488]))
_layer47 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer47.weight = torch.nn.Parameter(torch.tensor([[-0.7226057052612305]]).transpose(-2, -1))
    _layer47.bias = torch.nn.Parameter(torch.tensor([-0.3567482829093933]))
_layer48 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer48.weight = torch.nn.Parameter(torch.tensor([[0.5033785104751587]]).transpose(-2, -1))
    _layer48.bias = torch.nn.Parameter(torch.tensor([0.5915882587432861]))
_layer49 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer49.weight = torch.nn.Parameter(torch.tensor([[0.16688144207000732]]).transpose(-2, -1))
    _layer49.bias = torch.nn.Parameter(torch.tensor([-0.27440404891967773]))
_layer50 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer50.weight = torch.nn.Parameter(torch.tensor([[0.03039097785949707]]).transpose(-2, -1))
    _layer50.bias = torch.nn.Parameter(torch.tensor([0.6279025077819824]))
_layer51 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer51.weight = torch.nn.Parameter(torch.tensor([[0.3989309072494507]]).transpose(-2, -1))
    _layer51.bias = torch.nn.Parameter(torch.tensor([-0.6781611442565918]))
_layer52 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer52.weight = torch.nn.Parameter(torch.tensor([[0.32473480701446533]]).transpose(-2, -1))
    _layer52.bias = torch.nn.Parameter(torch.tensor([0.9541763067245483]))
_layer53 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer53.weight = torch.nn.Parameter(torch.tensor([[-0.4985625147819519]]).transpose(-2, -1))
    _layer53.bias = torch.nn.Parameter(torch.tensor([0.8650431632995605]))
tuple10 = torch.nn.Sequential(
    _layer44,
    _layer45,
    _layer46,
    _layer47,
    _layer48,
    _layer49,
    _layer50,
    _layer51,
    _layer52,
    _layer53,
)
_layer54 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer54.weight = torch.nn.Parameter(torch.tensor([[-0.3783159852027893]]).transpose(-2, -1))
    _layer54.bias = torch.nn.Parameter(torch.tensor([0.40431857109069824]))
_layer55 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer55.weight = torch.nn.Parameter(torch.tensor([[0.4027813673019409]]).transpose(-2, -1))
    _layer55.bias = torch.nn.Parameter(torch.tensor([0.8798936605453491]))
_layer56 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer56.weight = torch.nn.Parameter(torch.tensor([[0.4876202344894409]]).transpose(-2, -1))
    _layer56.bias = torch.nn.Parameter(torch.tensor([0.4455273151397705]))
_layer57 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer57.weight = torch.nn.Parameter(torch.tensor([[0.6076211929321289]]).transpose(-2, -1))
    _layer57.bias = torch.nn.Parameter(torch.tensor([0.9715057611465454]))
_layer58 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer58.weight = torch.nn.Parameter(torch.tensor([[0.29230356216430664]]).transpose(-2, -1))
    _layer58.bias = torch.nn.Parameter(torch.tensor([-0.6545597314834595]))
_layer59 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer59.weight = torch.nn.Parameter(torch.tensor([[-0.18756628036499023]]).transpose(-2, -1))
    _layer59.bias = torch.nn.Parameter(torch.tensor([-0.10851645469665527]))
_layer60 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer60.weight = torch.nn.Parameter(torch.tensor([[-0.6491999626159668]]).transpose(-2, -1))
    _layer60.bias = torch.nn.Parameter(torch.tensor([0.16038250923156738]))
_layer61 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer61.weight = torch.nn.Parameter(torch.tensor([[-0.9243364334106445]]).transpose(-2, -1))
    _layer61.bias = torch.nn.Parameter(torch.tensor([0.9195624589920044]))
_layer62 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer62.weight = torch.nn.Parameter(torch.tensor([[-0.13484114408493042]]).transpose(-2, -1))
    _layer62.bias = torch.nn.Parameter(torch.tensor([-0.05838203430175781]))
_layer63 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer63.weight = torch.nn.Parameter(torch.tensor([[-0.03340524435043335]]).transpose(-2, -1))
    _layer63.bias = torch.nn.Parameter(torch.tensor([0.8109714984893799]))
_layer64 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer64.weight = torch.nn.Parameter(torch.tensor([[-0.847286581993103]]).transpose(-2, -1))
    _layer64.bias = torch.nn.Parameter(torch.tensor([0.7178159952163696]))
tuple11 = torch.nn.Sequential(
    _layer54,
    _layer55,
    _layer56,
    _layer57,
    _layer58,
    _layer59,
    _layer60,
    _layer61,
    _layer62,
    _layer63,
    _layer64,
)
_layer65 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer65.weight = torch.nn.Parameter(torch.tensor([[-0.20273202657699585]]).transpose(-2, -1))
    _layer65.bias = torch.nn.Parameter(torch.tensor([-0.2639928460121155]))
_layer66 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer66.weight = torch.nn.Parameter(torch.tensor([[-0.46112722158432007]]).transpose(-2, -1))
    _layer66.bias = torch.nn.Parameter(torch.tensor([-0.6156734824180603]))
_layer67 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer67.weight = torch.nn.Parameter(torch.tensor([[0.4586758613586426]]).transpose(-2, -1))
    _layer67.bias = torch.nn.Parameter(torch.tensor([0.6268856525421143]))
_layer68 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer68.weight = torch.nn.Parameter(torch.tensor([[0.4483414888381958]]).transpose(-2, -1))
    _layer68.bias = torch.nn.Parameter(torch.tensor([0.36446428298950195]))
_layer69 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer69.weight = torch.nn.Parameter(torch.tensor([[-0.9419503808021545]]).transpose(-2, -1))
    _layer69.bias = torch.nn.Parameter(torch.tensor([0.24994564056396484]))
_layer70 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer70.weight = torch.nn.Parameter(torch.tensor([[0.9953192472457886]]).transpose(-2, -1))
    _layer70.bias = torch.nn.Parameter(torch.tensor([-0.6054648756980896]))
_layer71 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer71.weight = torch.nn.Parameter(torch.tensor([[0.6203018426895142]]).transpose(-2, -1))
    _layer71.bias = torch.nn.Parameter(torch.tensor([0.6848907470703125]))
_layer72 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer72.weight = torch.nn.Parameter(torch.tensor([[-0.4301600456237793]]).transpose(-2, -1))
    _layer72.bias = torch.nn.Parameter(torch.tensor([0.7356255054473877]))
_layer73 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer73.weight = torch.nn.Parameter(torch.tensor([[0.7850341796875]]).transpose(-2, -1))
    _layer73.bias = torch.nn.Parameter(torch.tensor([0.32139527797698975]))
_layer74 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer74.weight = torch.nn.Parameter(torch.tensor([[0.6747739315032959]]).transpose(-2, -1))
    _layer74.bias = torch.nn.Parameter(torch.tensor([0.8503944873809814]))
_layer75 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer75.weight = torch.nn.Parameter(torch.tensor([[0.2769254446029663]]).transpose(-2, -1))
    _layer75.bias = torch.nn.Parameter(torch.tensor([-0.5336059331893921]))
_layer76 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer76.weight = torch.nn.Parameter(torch.tensor([[-0.41154956817626953]]).transpose(-2, -1))
    _layer76.bias = torch.nn.Parameter(torch.tensor([-0.9820307493209839]))
tuple12 = torch.nn.Sequential(
    _layer65,
    _layer66,
    _layer67,
    _layer68,
    _layer69,
    _layer70,
    _layer71,
    _layer72,
    _layer73,
    _layer74,
    _layer75,
    _layer76,
)
_layer77 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer77.weight = torch.nn.Parameter(torch.tensor([[0.9023672342300415]]).transpose(-2, -1))
    _layer77.bias = torch.nn.Parameter(torch.tensor([-0.8075411915779114]))
_layer78 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer78.weight = torch.nn.Parameter(torch.tensor([[-0.5062331557273865]]).transpose(-2, -1))
    _layer78.bias = torch.nn.Parameter(torch.tensor([0.6352475881576538]))
_layer79 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer79.weight = torch.nn.Parameter(torch.tensor([[0.11726391315460205]]).transpose(-2, -1))
    _layer79.bias = torch.nn.Parameter(torch.tensor([-0.15892380475997925]))
_layer80 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer80.weight = torch.nn.Parameter(torch.tensor([[0.38878488540649414]]).transpose(-2, -1))
    _layer80.bias = torch.nn.Parameter(torch.tensor([-0.7222588062286377]))
_layer81 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer81.weight = torch.nn.Parameter(torch.tensor([[0.667579174041748]]).transpose(-2, -1))
    _layer81.bias = torch.nn.Parameter(torch.tensor([-0.8946297764778137]))
_layer82 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer82.weight = torch.nn.Parameter(torch.tensor([[-0.418423056602478]]).transpose(-2, -1))
    _layer82.bias = torch.nn.Parameter(torch.tensor([0.8719900846481323]))
_layer83 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer83.weight = torch.nn.Parameter(torch.tensor([[-0.7117000818252563]]).transpose(-2, -1))
    _layer83.bias = torch.nn.Parameter(torch.tensor([-0.4355848431587219]))
_layer84 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer84.weight = torch.nn.Parameter(torch.tensor([[-0.8798741102218628]]).transpose(-2, -1))
    _layer84.bias = torch.nn.Parameter(torch.tensor([0.34594857692718506]))
_layer85 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer85.weight = torch.nn.Parameter(torch.tensor([[0.7289218902587891]]).transpose(-2, -1))
    _layer85.bias = torch.nn.Parameter(torch.tensor([-0.5446890592575073]))
_layer86 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer86.weight = torch.nn.Parameter(torch.tensor([[-0.2579585313796997]]).transpose(-2, -1))
    _layer86.bias = torch.nn.Parameter(torch.tensor([-0.26795119047164917]))
_layer87 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer87.weight = torch.nn.Parameter(torch.tensor([[-0.1364964246749878]]).transpose(-2, -1))
    _layer87.bias = torch.nn.Parameter(torch.tensor([-0.634078323841095]))
_layer88 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer88.weight = torch.nn.Parameter(torch.tensor([[-0.7407867908477783]]).transpose(-2, -1))
    _layer88.bias = torch.nn.Parameter(torch.tensor([0.20381271839141846]))
_layer89 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer89.weight = torch.nn.Parameter(torch.tensor([[-0.3598601222038269]]).transpose(-2, -1))
    _layer89.bias = torch.nn.Parameter(torch.tensor([0.6859334707260132]))
tuple13 = torch.nn.Sequential(
    _layer77,
    _layer78,
    _layer79,
    _layer80,
    _layer81,
    _layer82,
    _layer83,
    _layer84,
    _layer85,
    _layer86,
    _layer87,
    _layer88,
    _layer89,
)
_layer90 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer90.weight = torch.nn.Parameter(torch.tensor([[0.1532588005065918]]).transpose(-2, -1))
    _layer90.bias = torch.nn.Parameter(torch.tensor([0.48228883743286133]))
_layer91 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer91.weight = torch.nn.Parameter(torch.tensor([[0.04923105239868164]]).transpose(-2, -1))
    _layer91.bias = torch.nn.Parameter(torch.tensor([-0.6121399402618408]))
_layer92 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer92.weight = torch.nn.Parameter(torch.tensor([[-0.8135586380958557]]).transpose(-2, -1))
    _layer92.bias = torch.nn.Parameter(torch.tensor([-0.8969761729240417]))
_layer93 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer93.weight = torch.nn.Parameter(torch.tensor([[-0.7959255576133728]]).transpose(-2, -1))
    _layer93.bias = torch.nn.Parameter(torch.tensor([-0.15254050493240356]))
_layer94 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer94.weight = torch.nn.Parameter(torch.tensor([[0.809080958366394]]).transpose(-2, -1))
    _layer94.bias = torch.nn.Parameter(torch.tensor([0.08910346031188965]))
_layer95 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer95.weight = torch.nn.Parameter(torch.tensor([[0.730097770690918]]).transpose(-2, -1))
    _layer95.bias = torch.nn.Parameter(torch.tensor([0.3269423246383667]))
_layer96 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer96.weight = torch.nn.Parameter(torch.tensor([[0.3307996988296509]]).transpose(-2, -1))
    _layer96.bias = torch.nn.Parameter(torch.tensor([-0.9675214886665344]))
_layer97 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer97.weight = torch.nn.Parameter(torch.tensor([[0.22882235050201416]]).transpose(-2, -1))
    _layer97.bias = torch.nn.Parameter(torch.tensor([-0.34953832626342773]))
_layer98 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer98.weight = torch.nn.Parameter(torch.tensor([[0.4173060655593872]]).transpose(-2, -1))
    _layer98.bias = torch.nn.Parameter(torch.tensor([0.3602440357208252]))
_layer99 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer99.weight = torch.nn.Parameter(torch.tensor([[0.45326507091522217]]).transpose(-2, -1))
    _layer99.bias = torch.nn.Parameter(torch.tensor([-0.596025824546814]))
_layer100 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer100.weight = torch.nn.Parameter(torch.tensor([[0.09676134586334229]]).transpose(-2, -1))
    _layer100.bias = torch.nn.Parameter(torch.tensor([-0.1919446587562561]))
_layer101 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer101.weight = torch.nn.Parameter(torch.tensor([[0.9914546012878418]]).transpose(-2, -1))
    _layer101.bias = torch.nn.Parameter(torch.tensor([0.2441469430923462]))
_layer102 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer102.weight = torch.nn.Parameter(torch.tensor([[0.1911848783493042]]).transpose(-2, -1))
    _layer102.bias = torch.nn.Parameter(torch.tensor([-0.8447966575622559]))
_layer103 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer103.weight = torch.nn.Parameter(torch.tensor([[-0.31683868169784546]]).transpose(-2, -1))
    _layer103.bias = torch.nn.Parameter(torch.tensor([0.14637601375579834]))
tuple14 = torch.nn.Sequential(
    _layer90,
    _layer91,
    _layer92,
    _layer93,
    _layer94,
    _layer95,
    _layer96,
    _layer97,
    _layer98,
    _layer99,
    _layer100,
    _layer101,
    _layer102,
    _layer103,
)
_layer104 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer104.weight = torch.nn.Parameter(torch.tensor([[0.6020323038101196]]).transpose(-2, -1))
    _layer104.bias = torch.nn.Parameter(torch.tensor([0.9433172941207886]))
_layer105 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer105.weight = torch.nn.Parameter(torch.tensor([[0.604312539100647]]).transpose(-2, -1))
    _layer105.bias = torch.nn.Parameter(torch.tensor([-0.8879148364067078]))
_layer106 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer106.weight = torch.nn.Parameter(torch.tensor([[0.4274866580963135]]).transpose(-2, -1))
    _layer106.bias = torch.nn.Parameter(torch.tensor([0.4936317205429077]))
_layer107 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer107.weight = torch.nn.Parameter(torch.tensor([[0.03321707248687744]]).transpose(-2, -1))
    _layer107.bias = torch.nn.Parameter(torch.tensor([0.23012447357177734]))
_layer108 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer108.weight = torch.nn.Parameter(torch.tensor([[-0.5317589640617371]]).transpose(-2, -1))
    _layer108.bias = torch.nn.Parameter(torch.tensor([-0.6021583676338196]))
_layer109 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer109.weight = torch.nn.Parameter(torch.tensor([[0.893057107925415]]).transpose(-2, -1))
    _layer109.bias = torch.nn.Parameter(torch.tensor([0.8639611005783081]))
_layer110 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer110.weight = torch.nn.Parameter(torch.tensor([[0.5530872344970703]]).transpose(-2, -1))
    _layer110.bias = torch.nn.Parameter(torch.tensor([-0.3874278664588928]))
_layer111 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer111.weight = torch.nn.Parameter(torch.tensor([[-0.43021392822265625]]).transpose(-2, -1))
    _layer111.bias = torch.nn.Parameter(torch.tensor([-0.7852112650871277]))
_layer112 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer112.weight = torch.nn.Parameter(torch.tensor([[0.07894527912139893]]).transpose(-2, -1))
    _layer112.bias = torch.nn.Parameter(torch.tensor([0.37543153762817383]))
_layer113 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer113.weight = torch.nn.Parameter(torch.tensor([[-0.21215307712554932]]).transpose(-2, -1))
    _layer113.bias = torch.nn.Parameter(torch.tensor([0.22721374034881592]))
_layer114 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer114.weight = torch.nn.Parameter(torch.tensor([[-0.8172506093978882]]).transpose(-2, -1))
    _layer114.bias = torch.nn.Parameter(torch.tensor([-0.18335551023483276]))
_layer115 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer115.weight = torch.nn.Parameter(torch.tensor([[0.06533777713775635]]).transpose(-2, -1))
    _layer115.bias = torch.nn.Parameter(torch.tensor([0.5855649709701538]))
_layer116 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer116.weight = torch.nn.Parameter(torch.tensor([[0.20346713066101074]]).transpose(-2, -1))
    _layer116.bias = torch.nn.Parameter(torch.tensor([0.6481508016586304]))
_layer117 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer117.weight = torch.nn.Parameter(torch.tensor([[0.40866339206695557]]).transpose(-2, -1))
    _layer117.bias = torch.nn.Parameter(torch.tensor([-0.3727037310600281]))
_layer118 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer118.weight = torch.nn.Parameter(torch.tensor([[-0.8936826586723328]]).transpose(-2, -1))
    _layer118.bias = torch.nn.Parameter(torch.tensor([0.3034778833389282]))
tuple15 = torch.nn.Sequential(
    _layer104,
    _layer105,
    _layer106,
    _layer107,
    _layer108,
    _layer109,
    _layer110,
    _layer111,
    _layer112,
    _layer113,
    _layer114,
    _layer115,
    _layer116,
    _layer117,
    _layer118,
)
_layer119 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer119.weight = torch.nn.Parameter(torch.tensor([[0.7879606485366821]]).transpose(-2, -1))
    _layer119.bias = torch.nn.Parameter(torch.tensor([0.6377499103546143]))
_layer120 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer120.weight = torch.nn.Parameter(torch.tensor([[0.7194042205810547]]).transpose(-2, -1))
    _layer120.bias = torch.nn.Parameter(torch.tensor([0.02915787696838379]))
_layer121 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer121.weight = torch.nn.Parameter(torch.tensor([[0.198927640914917]]).transpose(-2, -1))
    _layer121.bias = torch.nn.Parameter(torch.tensor([-0.5820839405059814]))
_layer122 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer122.weight = torch.nn.Parameter(torch.tensor([[0.4340386390686035]]).transpose(-2, -1))
    _layer122.bias = torch.nn.Parameter(torch.tensor([0.605263352394104]))
_layer123 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer123.weight = torch.nn.Parameter(torch.tensor([[-0.47579288482666016]]).transpose(-2, -1))
    _layer123.bias = torch.nn.Parameter(torch.tensor([-0.6321081519126892]))
_layer124 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer124.weight = torch.nn.Parameter(torch.tensor([[0.15942323207855225]]).transpose(-2, -1))
    _layer124.bias = torch.nn.Parameter(torch.tensor([0.898227334022522]))
_layer125 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer125.weight = torch.nn.Parameter(torch.tensor([[-0.46525347232818604]]).transpose(-2, -1))
    _layer125.bias = torch.nn.Parameter(torch.tensor([0.13947296142578125]))
_layer126 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer126.weight = torch.nn.Parameter(torch.tensor([[-0.2838357090950012]]).transpose(-2, -1))
    _layer126.bias = torch.nn.Parameter(torch.tensor([-0.19179344177246094]))
_layer127 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer127.weight = torch.nn.Parameter(torch.tensor([[-0.33708977699279785]]).transpose(-2, -1))
    _layer127.bias = torch.nn.Parameter(torch.tensor([0.09504163265228271]))
_layer128 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer128.weight = torch.nn.Parameter(torch.tensor([[0.42440342903137207]]).transpose(-2, -1))
    _layer128.bias = torch.nn.Parameter(torch.tensor([-0.848053514957428]))
_layer129 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer129.weight = torch.nn.Parameter(torch.tensor([[-0.9990988373756409]]).transpose(-2, -1))
    _layer129.bias = torch.nn.Parameter(torch.tensor([0.433229923248291]))
_layer130 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer130.weight = torch.nn.Parameter(torch.tensor([[0.1953490972518921]]).transpose(-2, -1))
    _layer130.bias = torch.nn.Parameter(torch.tensor([-0.4806498885154724]))
_layer131 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer131.weight = torch.nn.Parameter(torch.tensor([[-0.5913803577423096]]).transpose(-2, -1))
    _layer131.bias = torch.nn.Parameter(torch.tensor([-0.171758234500885]))
_layer132 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer132.weight = torch.nn.Parameter(torch.tensor([[0.17869138717651367]]).transpose(-2, -1))
    _layer132.bias = torch.nn.Parameter(torch.tensor([-0.3985886573791504]))
_layer133 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer133.weight = torch.nn.Parameter(torch.tensor([[0.14870023727416992]]).transpose(-2, -1))
    _layer133.bias = torch.nn.Parameter(torch.tensor([-0.9853461980819702]))
_layer134 = torch.nn.Linear(1, 1)
with torch.no_grad():
    _layer134.weight = torch.nn.Parameter(torch.tensor([[0.906070351600647]]).transpose(-2, -1))
    _layer134.bias = torch.nn.Parameter(torch.tensor([-0.38755929470062256]))
tuple16 = torch.nn.Sequential(
    _layer119,
    _layer120,
    _layer121,
    _layer122,
    _layer123,
    _layer124,
    _layer125,
    _layer126,
    _layer127,
    _layer128,
    _layer129,
    _layer130,
    _layer131,
    _layer132,
    _layer133,
    _layer134,
)

__all__ = []
