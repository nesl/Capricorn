#include <stdlib.h>
#include <stdio.h>
#include <math.h>

class DrillClassifier {
    public:
    static int drill_predict (double features[]) {
        int N_FEATURES = 32;
        int  N_CLASSES =2;
        int  N_VECTORS =42;
        int N_ROWS =2;
        int N_COEFFICIENTS =1;
        int N_INTERCEPTS =1;
        char KERNEL_TYPE ='r';
        double KERNEL_GAMMA =0.30659118168121746;
        double KERNEL_COEF =0.0;
        int KERNEL_DEGREE =3;                           

    double vectors[42][32] = {{0.0, 0.011910900848379239, 0.8312876739362149, 0.7407891289132084, 0.6220560706131545, 0.5791198211758548, 0.6366445169880738, 0.7446990037997696, 0.45015263306234476, 0.5588543417482381, 0.6291389988352416, 0.4990072445237169, 0.440923241979699, 1.0, 0.6006181858732036, 0.5013575264001713, 0.8963407542774708, 0.6935638022902417, 0.6008542762206489, 0.6348999174308908, 0.6388870099061179, 0.6301646277518614, 0.7677162483044195, 0.8286797522316263, 0.5215133345534136, 0.42686093766479805, 0.5233496092942923, 0.5863757272841945, 0.48786042141935126, 0.5622307610985615, 0.5722981479296114, 0.648300947617324}, {0.0, 0.01098736437905618, 0.3713083358785974, 0.6576070255458006, 0.6358148789222067, 0.6085037147810836, 0.354715783398509, 0.37217855173837344, 0.5523484075238108, 0.6356366546444782, 0.5295552872772658, 0.47165931569491293, 0.5991663632413589, 0.5357063590269914, 0.5879276582454106, 0.7326944786069394, 0.5934213058223666, 0.7678345874847043, 0.681645726731927, 0.8547002968860401, 0.5445631031464846, 1.0, 0.6541138827802555, 0.543728739653564, 0.4971477561616055, 0.3779865124806108, 0.40812106548625027, 0.3764669313587274, 0.5798957891090566, 0.3513391023331293, 0.589136218149152, 0.7740047430098556}, {0.0, 0.009616341359441168, 0.5044824856053562, 0.5130659289415929, 0.5653406946257045, 0.42651759993535443, 1.0, 0.5260193045912139, 0.42399662197088794, 0.4824153963292466, 0.7678422521095468, 0.2682148569850779, 0.5249892189249243, 0.42706071927540906, 0.3819145830797163, 0.4764708153063234, 0.532516566251015, 0.5404227065380204, 0.6580319550804145, 0.38928306821402187, 0.3555005455913667, 0.43227055104335127, 0.5444473727367556, 0.4893857119642435, 0.36608084855553324, 0.6258026843624632, 0.6363701157157534, 0.4925067053413262, 0.5211311067232324, 0.4678418707308266, 0.5616088231090294, 0.46434873073871014}, {0.0, 0.025794833959273754, 0.3271066867825148, 0.7108810190346451, 0.7462661645550388, 0.3570890109542701, 0.541413604190003, 0.333015787671304, 0.5463342250099356, 0.39984385493413604, 0.47430526343165114, 0.4130083615328525, 0.5717189853255528, 0.5329876217834019, 1.0, 0.5236816652961875, 0.43414312926351173, 0.48590185048378604, 0.5189961852516091, 0.3914851412511013, 0.6646797399440617, 0.4907032213626355, 0.36042291159150397, 0.41945894097081843, 0.46674529446419577, 0.4533794111755604, 0.5413691799557587, 0.46753399418813035, 0.4952636513667621, 0.5077008737325123, 0.5453872559637123, 0.5713767999774528}, {0.0, 0.010748895191370907, 0.9394986886080646, 0.6018366525601145, 0.7900598548029611, 0.6023687703787131, 0.6150148304443042, 0.9071561557988077, 0.47841255581037023, 1.0, 0.6106309126063265, 0.648983203793317, 0.971216562160256, 0.5309141925080333, 0.7774855168190083, 0.6749570494152342, 0.7137757759352144, 0.561038620590288, 0.5540553729565827, 0.5419879017078397, 0.5658255376180983, 0.5291678540199851, 0.5754467889611349, 0.6128567591575181, 0.7842561431815591, 0.6510614265972626, 0.48292265101033577, 0.5104132643387665, 0.6902141515312464, 0.659765996074725, 0.6279650049510229, 0.7223481176347035}, {0.0, 0.024144226340723572, 0.43308713079416294, 1.0, 0.4686695473481297, 0.583961771050633, 0.5469683498357663, 0.6073175678466505, 0.4049819256406139, 0.49155407816941726, 0.548316980675783, 0.4595742159281183, 0.4306726545284088, 0.5535821008512379, 0.3982706942681392, 0.320634946236279, 0.5222670677770131, 0.6881392680139526, 0.5636455906045742, 0.5290147654456531, 0.44263737749298365, 0.3999904803959418, 0.4427649503558824, 0.3669059589842915, 0.4827342202261946, 0.5769619108244812, 0.5535617099395016, 0.5940308732815491, 0.5665704034584322, 0.4520102759720443, 0.4420048307282125, 0.4109626789699155}, {0.0, 0.021326308975908568, 0.6135299018084529, 1.0, 0.5014782642367163, 0.545721077066144, 0.5158661538647467, 0.6298362033307409, 0.6428048688876815, 0.5935953124130701, 0.5699995981848099, 0.4893426822239987, 0.452332202831387, 0.6494519950469195, 0.5235979181068395, 0.368606268505057, 0.49821085177519125, 0.674606125362263, 0.5799011131754304, 0.5629605013374807, 0.515780712559051, 0.4548530145734832, 0.6687506534732294, 0.4728915788680641, 0.5359130197000356, 0.6087932588099456, 0.8344672320398564, 0.5642636146467225, 0.553422852931544, 0.5210449306564711, 0.4473400663980996, 0.38317367944837105}, {0.0, 0.0634243292166536, 0.3418716443368951, 0.7506052095484087, 0.3855196524309192, 0.4876825055120684, 0.49849358543211053, 0.4182894132355902, 0.334345557255915, 0.538287686573349, 0.43116708907216994, 0.48249236807739754, 0.4063048308879544, 0.43888728544410793, 0.5763443269416596, 0.4770464125787869, 0.2750049447531152, 0.3878773578793331, 0.4490085861283675, 0.5123616827694101, 0.4204174030849809, 0.39428177517856855, 0.5516825527963872, 0.2802718039811891, 0.4359358379143858, 0.2736342424227857, 0.3911349347167505, 0.37045159779631015, 0.6938233399659678, 0.3092159719993113, 0.49649383564815897, 1.0}, {0.0, 0.0283377965993876, 0.32080316200050235, 0.47064488348140643, 0.323349883837281, 0.6499313261890453, 0.44108809586023867, 0.6405650794173205, 0.3716710373680419, 0.48452215292631057, 0.5595117040664469, 0.525773025435215, 0.32530459036303866, 0.46868502567026354, 0.4887870955070753, 0.4816499690381192, 0.3192007062134949, 0.34857144167791126, 0.44530754542835477, 0.5255164890854427, 0.4629420684117644, 0.5117789492408147, 0.53242680963064, 0.42106096133752646, 0.5611498165599974, 0.3846569233678882, 0.42418644116719323, 0.4943485685615617, 0.5436110386945348, 0.3939041304706194, 0.4403563233789306, 1.0}, {0.0, 0.0340711978942329, 0.33295084545290743, 0.4875256030194944, 0.45583512445688606, 0.5320175281685714, 0.42005376921204557, 0.5477386862660119, 0.4785265649153101, 0.5243842466257169, 0.6088231178458565, 0.4179085913107386, 0.4852496468924406, 0.4456732913642005, 0.6194803729849802, 0.495632750208478, 0.3956465452092184, 0.4071234889097648, 0.35587677973667237, 0.43621834294744866, 0.42129524945353847, 0.5679907808201888, 0.4915955820220786, 0.27735804107227147, 0.5199084003721249, 0.4417463766217601, 0.48630066914961007, 0.4602640330074482, 0.4538321206532377, 0.47719402066473715, 0.32742983933153874, 1.0}, {0.0, 0.028728661544896304, 0.35546147377359316, 1.0, 0.3809324021263521, 0.42300060668178263, 0.6065227722690736, 0.5457606951596203, 0.4566302424958387, 0.5041222453560245, 0.4248218659990553, 0.33693256905681346, 0.4611373075163521, 0.5149123341081627, 0.3869238267560807, 0.37056299959218847, 0.37154194926514134, 0.48599521861072875, 0.5667299977793229, 0.37223379404165186, 0.38409212133798437, 0.4092521752099272, 0.3788714854807531, 0.4155278459363953, 0.45599661708586614, 0.7082906732412797, 0.3148494320144793, 0.40332072486596976, 0.40576852109479694, 0.3556669507582386, 0.39090911031341097, 0.3283224842831049}, {0.0, 0.03589840491284592, 0.48112375322671347, 0.7345357198702019, 0.6391248663541363, 1.0, 0.4270315480779471, 0.7810611670179368, 0.4069759803250663, 0.4931562267736817, 0.6924588045572724, 0.5599077780758321, 0.46266414692467495, 0.6042411202716389, 0.46594700969398745, 0.6990556032962855, 0.3709725011559124, 0.43709325111155445, 0.6800167452929868, 0.5710145539151156, 0.5684254891501158, 0.4750053421893114, 0.5500969260955866, 0.4549642807574811, 0.5117030982197353, 0.5493587244290388, 0.5471990546381486, 0.603887392404548, 0.4879676653844006, 0.419522737592029, 0.5489689035429312, 0.49276526543958005}, {0.0, 0.024756664950025806, 0.4293443206255388, 1.0, 0.6571543714731288, 0.8145453828955536, 0.42091656610802525, 0.7475395497854175, 0.4788353252329291, 0.4722951579763774, 0.5849485824026499, 0.611492365417178, 0.6112154150043002, 0.6063318042925373, 0.5838708592243879, 0.5094436177171681, 0.4239622603359186, 0.4987273188224747, 0.781610088057741, 0.4613441659240049, 0.5165936216316537, 0.5189859149316177, 0.5245561331518053, 0.4471618826989402, 0.43537838117222805, 0.5246603232369081, 0.4101457352914375, 0.5921024827017147, 0.4563433842046012, 0.4452614719702994, 0.5847685690322477, 0.503908705065834}, {0.0, 0.052060335897774423, 0.3669753736241351, 0.46938963326987787, 0.6074522257053437, 0.45926339258413235, 0.45810520909696606, 0.4672119373421356, 0.4828367290964269, 0.6292490622197593, 0.4970874723520853, 0.43901931029678215, 0.4789236061176946, 0.38679683782980123, 0.435259753498227, 0.532368485672004, 0.43757863834725036, 0.5007747778063962, 1.0, 0.5572599124376219, 0.4930741201733899, 0.6221857594258242, 0.6379435447664141, 0.5287130883821183, 0.4875190530097759, 0.6021035104562612, 0.36483410974074537, 0.5141380750869774, 0.4459050531009794, 0.3942408458271342, 0.5869355432777883, 0.4444210040585231}, {0.0, 0.0006774747814781945, 0.07745002514179496, 0.6890807545977035, 1.0, 0.8063472403182421, 0.4262530843571557, 0.34809636557591767, 0.4426847323596805, 0.5479779159409622, 0.3393434216431248, 0.45997062788721416, 0.39572451777386475, 0.269300252342679, 0.46198600751127405, 0.40719194858964064, 0.47879553731437013, 0.5034763769234125, 0.5825037373855859, 0.4893881392242464, 0.42034979562266356, 0.34656351098699223, 0.45327427986019625, 0.43345788441910577, 0.4459655941967424, 0.36010297986447953, 0.4066107768252125, 0.5173145737470778, 0.3573375330565204, 0.4098733978929452, 0.3983013280678689, 0.3583907182585827}, {0.0, 0.005599401970576772, 0.06437942750462113, 0.8965612153792917, 0.9453105666918271, 1.0, 0.5087972448164148, 0.5650423498679709, 0.5525251462825183, 0.4458917064549724, 0.3386975213194376, 0.3981817582038161, 0.44713504577660645, 0.35791779531462303, 0.45243741396702375, 0.607386046650145, 0.4436848468924332, 0.43589963924534125, 0.5342292174709248, 0.5189244213559591, 0.4835060527330769, 0.4854127526278611, 0.5211555747416458, 0.5519064739736459, 0.39128798043052876, 0.5876679521047937, 0.40301603286745147, 0.5283054397664136, 0.4026202462385007, 0.49163418859901414, 0.47017914645507763, 0.482247114545832}, {0.0, 0.0013514753752894636, 0.012913744937308281, 0.1833396877082202, 0.4890804232103294, 0.6088851487219245, 1.0, 0.3489596816199576, 0.47446139834154577, 0.6227157909188599, 0.4763020860161048, 0.42244301847501636, 0.36729149786801474, 0.4094843282246932, 0.5529500141666801, 0.49024404275998296, 0.5143784777227876, 0.5627621576426937, 0.5615894181702916, 0.5426059071703079, 0.4067017067716778, 0.45014701555934145, 0.5245604727718338, 0.25093114037468517, 0.43719842257312985, 0.5577696275578988, 0.3376037479190578, 0.43930742706903936, 0.552864112665835, 0.3665795058050522, 0.3612675039605931, 0.4699817925777335}, {0.0, 0.027790942545422692, 0.7324182699862238, 0.6775376674772654, 0.6769271047977364, 0.617233439332, 0.6163923293703004, 0.6452182025664398, 0.5080662446211953, 1.0, 0.6868216944270851, 0.5072500243359244, 0.5163671189618116, 0.7858726165208545, 0.4233714852183705, 0.6263329365834623, 0.6627965121334993, 0.494384462676204, 0.6582768240083867, 0.6034112494588533, 0.594431891231271, 0.8416352789650526, 0.6182555176163175, 0.5474186246704819, 0.5592854105021299, 0.6631346005603969, 0.5346663277452695, 0.5769037284318825, 0.47917834597683984, 0.598033808712822, 0.482621137650766, 0.6106925602277896}, {0.0, 0.024215801665744406, 0.5053395642055133, 0.6295997033149952, 0.7527551564738555, 0.6009291831267066, 0.4839103907024588, 1.0, 0.41653867303848785, 0.6530743784188804, 0.4795490296549173, 0.5075570851519395, 0.6693652531983393, 0.5189792894005821, 0.5839712141562409, 0.44147688520836686, 0.4329208291785497, 0.5595277670691627, 0.5014483726041872, 0.4689411647146717, 0.4031407729577346, 0.49191739907220644, 0.6523820883179052, 0.6119712649374107, 0.46359169625404967, 0.40030805641129535, 0.6383841019128187, 0.561997589091422, 0.49634583688831824, 0.5456456077543245, 0.5202441192497465, 0.5413680614791256}, {0.0, 0.026010085296107297, 0.7664418086394436, 0.7369133955630501, 0.662779551145179, 0.72931143553102, 0.587719714868329, 1.0, 0.7574131315175134, 0.5530792327327118, 0.6179888486224723, 0.6767897526332011, 0.5986038892212461, 0.6676104229526622, 0.6414963001832164, 0.6103238811517735, 0.5157850897196792, 0.8795069260339414, 0.5432815664979211, 0.7957291638505221, 0.5159399691287104, 0.858133610392004, 0.5621378183753248, 0.5656999444349798, 0.5048595795802941, 0.6376323577713632, 0.4404264315654647, 0.7983731874452625, 0.5140961724134001, 0.6089665728499567, 0.5849462386228834, 0.48531651787177255}, {0.0, 0.005397876044302829, 0.35623156969976977, 0.6108513949095439, 0.9105129263100763, 0.42518254752289486, 0.6236979351484369, 0.5510684781935682, 0.5523178009437614, 0.4099161776030344, 0.5892763475257551, 0.6836173874326665, 0.6330345892803649, 0.47108268852186846, 0.5304604078530075, 0.5957289699126154, 0.3696577582944344, 0.41991747646885796, 0.5772403563500276, 0.9354791983257957, 0.557566209782821, 0.5697397724374436, 1.0, 0.6159304843725356, 0.509776459629744, 0.4870264654343598, 0.4580908528719155, 0.5356264384590466, 0.5463037097077275, 0.7232154344242973, 0.44871304225365133, 0.345824925721843}, {0.0, 0.022234171410720493, 0.6236910073103377, 0.6184056638287436, 0.6726050478764048, 0.5691903336815222, 0.7702631956085058, 0.8779595259588431, 0.705606558823903, 0.5345127865508423, 0.6958934158642576, 0.6346967664927542, 0.5932625218810644, 0.6630417550279907, 0.5574115229104921, 0.6925800783487347, 0.5484180405607024, 0.5719138077696364, 0.8266726243208726, 1.0, 0.4748024161315862, 0.5781140990132086, 0.48330146119950496, 0.5498793572760907, 0.6473132131240718, 0.7317485960988197, 0.5088050034579522, 0.4866003579077508, 0.578459432469337, 0.8565314738633543, 0.5241366658778521, 0.5200794295987435}, {0.0, 0.031739445768234044, 0.41302646458364933, 0.6641278949563048, 0.8230943212100993, 0.6147348376122362, 0.5757814349021649, 0.6400080440777189, 0.47461885145017696, 0.5088101337973887, 0.5271474305356225, 0.5670017656030494, 0.6011559942493335, 0.6335051660759425, 0.399942359166174, 0.4670884974582487, 0.5955336387940366, 0.4668187297897642, 0.5363708288026705, 1.0, 0.5907455668324251, 0.7985533878268567, 0.4199928228504499, 0.6637876509572586, 0.5468421171865776, 0.5526131780979507, 0.6187360410274395, 0.5052019320503993, 0.47200736926885845, 0.38767778959341487, 0.5558094991424867, 0.675479963919328}, {0.0, 0.01227768752637612, 0.4538596425129693, 0.6667398487803788, 0.6709197208602885, 0.5312296773595805, 0.35693912005723727, 0.6770980963050448, 0.541946429173161, 0.37957147350861276, 0.4381113678015289, 0.6221609661769264, 0.5230133508972107, 0.5677304849180901, 0.4516512236661782, 0.6231357740187433, 1.0, 0.4437424802542528, 0.5148538798618393, 0.486788122115583, 0.5219405641923842, 0.4677177321203493, 0.4856986577662441, 0.4434248446689618, 0.49821801683712585, 0.6271486679672231, 0.6334987993785504, 0.5721195587318704, 0.40116930691398917, 0.499131852734447, 0.49744404820553967, 0.46897925392978584}, {0.0, 0.012604055252594589, 0.590499402988757, 1.0, 0.39965643843061693, 0.31998764276298775, 0.42170976027093676, 0.36983889828854044, 0.28237766177152895, 0.41048804509890274, 0.35188562130340306, 0.31951431513825257, 0.4696589803954514, 0.3069902217562262, 0.3432259313624146, 0.3063826608876035, 0.2989116093125397, 0.463246631358324, 0.36934305010239066, 0.2654310448554598, 0.43818877060308925, 0.32170451305792747, 0.40546979448949966, 0.29072078843104227, 0.3340183917054427, 0.3386600448473806, 0.3899901640784695, 0.47282928738441027, 0.329984577073819, 0.37869135811876403, 0.29563915323598633, 0.48657663070118234}, {0.0, 0.036454470730980626, 0.6138938725148334, 1.0, 0.45165624520352377, 0.42020402599315637, 0.4917948409719874, 0.48989278722594515, 0.37324588619731747, 0.42750567632517866, 0.3070208898302896, 0.3146299594481273, 0.2820194318339353, 0.2952263487263075, 0.3988249562931767, 0.3367470733229655, 0.43407818996460706, 0.4103545857034511, 0.40813670824675347, 0.45178683866511987, 0.4279457512347564, 0.4219235095716585, 0.4051566167131218, 0.20812281902142002, 0.3984695984223373, 0.29164887518482374, 0.43766525507551673, 0.4302192084485353, 0.3506524427792568, 0.31801335657556007, 0.36635684984850636, 0.49741311645216785}, {0.0, 0.0246643911557854, 0.5676547685850147, 1.0, 0.4982212400034375, 0.3734519760719681, 0.31887228525349803, 0.4665989944428757, 0.45687496372652137, 0.4089374818039403, 0.3764479848460833, 0.304495625730908, 0.5637590186215852, 0.39293364724981683, 0.3684540758092023, 0.4477060438700912, 0.4147205053368539, 0.30866471912475546, 0.5064549915986664, 0.4423468771649247, 0.30375448636195507, 0.41494422095657196, 0.34965371776402043, 0.3875691166296768, 0.3708404617650282, 0.4022664798390949, 0.512459645973788, 0.424287630734606, 0.32568887954431947, 0.29806869029325017, 0.35484035399550634, 0.40236243375395797}, {0.0, 0.010365977290388908, 0.2334877735434372, 0.5963047075422292, 0.48498045234646514, 1.0, 0.44418019639184075, 0.4625540237337913, 0.37125599290111827, 0.2897353119643638, 0.20715361079127553, 0.29384511595227614, 0.4215290548690072, 0.3828921606260228, 0.4089353643366576, 0.4458354483666332, 0.4163515109746862, 0.3038598571422836, 0.3064386294800916, 0.31315638142288144, 0.30586419079902505, 0.33304109136418825, 0.31907495326048857, 0.1856231673532782, 0.225532599285248, 0.30487403341968766, 0.31834194962277884, 0.22476941033901093, 0.32741872007895967, 0.30007806346772953, 0.3573600617254383, 0.4785629259898558}, {0.0, 0.016134706384287142, 0.3710278725473097, 1.0, 0.7020771147087265, 0.7535176611491484, 0.6815831876509837, 0.5065564302022022, 0.6011112510354621, 0.47384505413173494, 0.3995390529827765, 0.36613184232875867, 0.31051501353532596, 0.2970321176562315, 0.4852761651024339, 0.5906289808134398, 0.2916737325267652, 0.2392256633928365, 0.3980765370560495, 0.541956242071924, 0.3637128922172565, 0.35957843838377723, 0.6423618993219707, 0.30263005170637286, 0.2764079218582792, 0.5033552800556067, 0.318839136783782, 0.3093998416378456, 0.34030597563343107, 0.29903319574041554, 0.3515626830547424, 0.5371313982404692}, {0.0, 0.015262472174833595, 0.7532979439587467, 0.6548741954106515, 0.8033028753608097, 1.0, 0.381211725916586, 0.7714525975879084, 0.5659888619511276, 0.48755522032902227, 0.42947781195527357, 0.3878234889367288, 0.4978819447054102, 0.5008350326072841, 0.518480537383263, 0.52432225213599, 0.4653638324642397, 0.47205149305433897, 0.3741124967622733, 0.5348113776182734, 0.5468065093417295, 0.39467618565558105, 0.33382709587160925, 0.38883940560549324, 0.4381013886997196, 0.6727539535803192, 0.3514989573613, 0.2648823847990284, 0.39646822751309074, 0.3450540414465973, 0.35909249360278594, 0.29995992822478434}, {0.0, 0.010051731665600765, 0.8345662064761461, 1.0, 0.975285137346731, 0.7195174092585642, 0.4108578596918171, 0.9956215572661214, 0.3466542787663488, 0.48776302219767714, 0.525059953899952, 0.5072465099249134, 0.41281658842655, 0.3773751515246226, 0.42593151951558234, 0.5583491013096291, 0.4520459142453341, 0.4447766807833996, 0.5297966623538535, 0.5234554123267355, 0.5977151069363569, 0.3799026010383648, 0.5749328265513732, 0.5813911604998044, 0.38547479426147413, 0.6295326877989011, 0.3457133783394257, 0.4707962186969536, 0.48681994944831447, 0.3937333900529085, 0.40543559278897695, 0.304595693331551}, {0.0, 0.025282349315824796, 0.8146157046225292, 0.8569937880056948, 0.7324486914174535, 0.6107216844568729, 0.42470809355571737, 1.0, 0.35307916772732223, 0.5499992025427776, 0.49346243503296966, 0.52925731512121, 0.3695816807257623, 0.4134459535073025, 0.4233763322693537, 0.5986361846579789, 0.4735091339763487, 0.3584255755311081, 0.3234009495292845, 0.5444405514246013, 0.4570025807375582, 0.4945404651308135, 0.5564295159708962, 0.41864127939291285, 0.3750662213776846, 0.39891544654863276, 0.29690845711937036, 0.5286631192840732, 0.5401254096537303, 0.36154235034526694, 0.4709771125137552, 0.329478954217772}, {0.0, 0.013945709532005338, 0.8087353193104071, 1.0, 0.5940167090614736, 0.8571074306778634, 0.5156394937240583, 0.9060587251731955, 0.3931024648743358, 0.4590068475728518, 0.49765530613208286, 0.3588617989799987, 0.5495222579746117, 0.40065550338106365, 0.45181459053640094, 0.6321469003201707, 0.4523177643925088, 0.37171294486873285, 0.37694936372775045, 0.4324859223023019, 0.48911131682257025, 0.4445563882366996, 0.5353372202304751, 0.4554752210844744, 0.35005287470817836, 0.4589153795260052, 0.35177014818218033, 0.4496216073113483, 0.4205990070305253, 0.3270140499343535, 0.3385809374907682, 0.2629077550743857}, {0.0, 0.004821654353991434, 0.4342658029479221, 0.4945845831617944, 0.6578271019819933, 0.515670727439326, 0.46590670204631945, 0.35553883519910784, 0.24921053740053314, 0.5810705836392491, 0.35738071427886614, 0.3779495799390397, 0.2563562761143294, 0.4278607764863658, 0.2948913493203156, 0.5826693975986502, 0.4029504019590472, 0.5773663865056065, 0.6196412661686489, 1.0, 0.3540750498520894, 0.4147528860745838, 0.3518319782473019, 0.5036431158976811, 0.323726495007213, 0.41786799257180524, 0.42794114665858507, 0.3220226662407987, 0.28626080089517947, 0.2856363419474243, 0.34159954015780625, 0.40932543292105233}, {0.0, 0.02664150601402184, 0.4988231477749966, 0.47375433515341886, 0.7584175571411034, 0.5550560573818303, 0.4816973799543, 0.34339063265900116, 0.5417998550433923, 0.6506888045034297, 0.4100867244715167, 0.47598347698105026, 0.3554516106110015, 0.5553380434160963, 0.29635077205962923, 0.44862484718618045, 0.3573679163792714, 0.6771478196277565, 0.8198160147098057, 1.0, 0.4085251048412214, 0.4377053978278157, 0.3740944391221605, 0.5467601171485192, 0.5819483610934053, 0.36565055770056376, 0.3923201844818612, 0.32008885994733727, 0.3783146320526781, 0.3996280288894131, 0.41070212514833404, 0.5504219766965858}, {0.0, 0.05181371098658134, 0.6548831446151351, 0.7427126983137009, 0.7268479610283466, 0.6898150058361969, 0.4544411480697298, 0.5145650076463323, 0.4710897775642515, 0.5463392135407178, 0.4145762325351577, 0.42140656978042446, 0.33758994418085475, 0.7786135357829157, 0.41971454118142204, 0.5033476113712796, 0.4456566935019968, 0.814853401276815, 1.0, 0.9272151685979773, 0.4563727537599845, 0.532364259304777, 0.5027570465261966, 0.5076366979295889, 0.6298186815027439, 0.5207229710770591, 0.39308193093471516, 0.331908279205836, 0.4376798864198545, 0.34835304748345114, 0.4584386110999559, 0.44385455551807856}, {0.0, 0.010769413661655543, 0.6848224094557982, 0.7663598553066633, 0.7428787720782631, 0.6536820268553297, 0.4611849137797806, 0.631200378510036, 0.4475806468649086, 0.5468876475617483, 0.35112425880794296, 0.5158044591598053, 0.4398129291359931, 0.8032019811801495, 0.40770338168378073, 0.42693688405654373, 0.43640087042774695, 0.48020349243513694, 1.0, 0.46922855772392114, 0.41814422400473766, 0.46033080829806555, 0.390817527225616, 0.40584319093238896, 0.4896975928661117, 0.5109188090348518, 0.39486904509693754, 0.3216481315634357, 0.4017313224682829, 0.35894040613228545, 0.3751691001475967, 0.3178965392651894}, {0.0, 0.03568916444683556, 0.6254975029113956, 1.0, 0.60879542611864, 0.7520739967465592, 0.600849314175569, 0.7715250090937235, 0.5226409977992056, 0.6170171813307557, 0.39677688121760213, 0.36625472286058, 0.5536996669433187, 0.7713554887469505, 0.5228191830212549, 0.583244751304145, 0.4001970379244515, 0.5724178878590935, 0.9189763659228123, 0.8530457985723591, 0.3820979796001247, 0.4307735501923263, 0.3391029570348412, 0.31309933800163153, 0.44669346621915856, 0.5611217156941923, 0.3311039463047728, 0.36083979966944235, 0.4238958965610969, 0.363973521745033, 0.39161633060933204, 0.31899901074700737}, {0.0, 0.024141253608397705, 0.7475538452607805, 1.0, 0.9844945234307789, 0.4985453953751665, 0.5322211834446356, 0.5778613844549402, 0.5174507311270592, 0.699003884003541, 0.7058303823329953, 0.4407750806332263, 0.6725342679537848, 0.5061030020179452, 0.711996784082065, 0.9090108125999931, 0.44416003508222557, 0.5070938915377348, 0.696785930161684, 0.9208779704511967, 0.5660620981098056, 0.46823351260733714, 0.4164783922540267, 0.39065722776198797, 0.4670547436934268, 0.8038636816425404, 0.3213836825754819, 0.3753715546416479, 0.3496306104756548, 0.4243983253309482, 0.3546281722336651, 0.4109513588647832}, {0.0, 0.05154337236929669, 0.9566316877862395, 0.4539509522439428, 1.0, 0.3943548719341362, 0.40508364770454486, 0.48969878157627994, 0.3667560297373392, 0.387251287705214, 0.5681601532290664, 0.36486687451336625, 0.34239667062965634, 0.4512228283059182, 0.502376348490859, 0.4728962216730965, 0.3829721412546059, 0.31041839311991093, 0.5358322850745166, 0.4774333134265342, 0.43088088549556364, 0.4438785986356247, 0.3245738800222926, 0.3008945777489641, 0.4520180599902343, 0.4425880657906795, 0.502021854413592, 0.4776901744776776, 0.42365897762532795, 0.40250615102711856, 0.6070933142744632, 0.32508290923334826}, {0.0, 0.052103486896994886, 1.0, 0.6527840409469173, 0.8916155076143802, 0.4689953971857486, 0.4527606848979619, 0.46386244839063834, 0.2752537020710499, 0.4475685562715729, 0.44204347117101195, 0.4148550940099657, 0.4239442149049614, 0.5587143948418672, 0.48517112256309847, 0.37369776237689045, 0.3954592294747178, 0.6590721627721455, 0.5558786771355375, 0.5054357925921467, 0.43229830973511557, 0.5962008288582961, 0.5206963010795688, 0.3795605686291057, 0.35896892513171935, 0.4498664239690641, 0.4781711732773743, 0.5049427997969056, 0.5195868946986221, 0.3434309007259581, 0.4352317114072822, 0.5479175444689804}, {0.0, 0.0008328733713923889, 0.019780818830788905, 0.0908513071242594, 1.0, 0.02790106188433782, 0.07361897810670406, 0.01987077043875326, 0.021821185272199416, 0.024665886240575013, 0.021229779038672993, 0.019326990187765965, 0.02043189733354396, 0.02449043522074882, 0.01563633317292372, 0.028314998231111905, 0.02225710302590092, 0.021982772288828806, 0.020987036947180796, 0.021476051334615744, 0.01630662042853144, 0.024223502088787922, 0.024746337533799623, 0.015441946947248939, 0.025724038087157328, 0.017690295045319545, 0.010721998325219282, 0.022800692365129927, 0.01876887149022508, 0.02036174232129439, 0.016506608599949495, 0.015512088569829965}};
    double coefficients[1][42] = {{-0.18555148429709292, -0.12451027046697616, -0.8434589044555233, -0.6119226572535102, -0.27051006636265035, -1.0, -1.0, -1.0, -0.23886737525734517, -0.36956811485773666, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.25358400998754055, -0.1842661851324789, -0.6202394043164277, -0.27591400883000333, -0.1608461550239612, -0.18972340439070726, -0.9686131567384927, -0.8639312691377351, 1.0, 0.4350871583275057, 1.0, 1.0, 1.0, 0.7523587404539549, 1.0, 1.0, 0.5188766012266297, 0.9326872010578472, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3216855483264341, 1.0, 0.20081121711581004}};
    double intercepts[1] = {-0.20592932036532274};
    int weights[2] = {24, 18};


    int i, j, k, d, l;

    double kernels[N_VECTORS];
    double kernel;
    switch (KERNEL_TYPE) {
        case 'l':
            // <x,x'>
            for (i = 0; i < N_VECTORS; i++) {
                kernel = 0.;
                for (j = 0; j < N_FEATURES; j++) {
                    kernel += vectors[i][j] * features[j];
                }
                kernels[i] = kernel;
            }
            break;
        case 'p':
            // (y<x,x'>+r)^d
            for (i = 0; i < N_VECTORS; i++) {
                kernel = 0.;
                for (j = 0; j < N_FEATURES; j++) {
                    kernel += vectors[i][j] * features[j];
                }
                kernels[i] = pow((KERNEL_GAMMA * kernel) + KERNEL_COEF, KERNEL_DEGREE);
            }
            break;
        case 'r':
            // exp(-y|x-x'|^2)
            for (i = 0; i < N_VECTORS; i++) {
                kernel = 0.;
                for (j = 0; j < N_FEATURES; j++) {
                    kernel += pow(vectors[i][j] - features[j], 2);
                }
                kernels[i] = exp(-KERNEL_GAMMA * kernel);
            }
            break;
        case 's':
            // tanh(y<x,x'>+r)
            for (i = 0; i < N_VECTORS; i++) {
                kernel = 0.;
                for (j = 0; j < N_FEATURES; j++) {
                    kernel += vectors[i][j] * features[j];
                }
                kernels[i] = tanh((KERNEL_GAMMA * kernel) + KERNEL_COEF);
            }
            break;
    }

    int starts[N_ROWS];
    int start;
    for (i = 0; i < N_ROWS; i++) {
        if (i != 0) {
            start = 0;
            for (j = 0; j < i; j++) {
                start += weights[j];
            }
            starts[i] = start;
        } else {
            starts[0] = 0;
        }
    }

    int ends[N_ROWS];
    for (i = 0; i < N_ROWS; i++) {
        ends[i] = weights[i] + starts[i];
    }

    if (N_CLASSES == 2) {

        for (i = 0; i < N_VECTORS; i++) {
            kernels[i] = -kernels[i];
        }

        double decision = 0.;
        for (k = starts[1]; k < ends[1]; k++) {
            decision += kernels[k] * coefficients[0][k];
        }
        for (k = starts[0]; k < ends[0]; k++) {
            decision += kernels[k] * coefficients[0][k];
        }
        decision += intercepts[0];

        if (decision > 0) {
            return 0;
        }
        return 1;

    }

    double decisions[N_INTERCEPTS];
    double tmp;
    for (i = 0, d = 0, l = N_ROWS; i < l; i++) {
        for (j = i + 1; j < l; j++) {
            tmp = 0.;
            for (k = starts[j]; k < ends[j]; k++) {
                tmp += kernels[k] * coefficients[i][k];
            }
            for (k = starts[i]; k < ends[i]; k++) {
                tmp += kernels[k] * coefficients[j - 1][k];
            }
            decisions[d] = tmp + intercepts[d];
            d = d + 1;
        }
    }

    int votes[N_INTERCEPTS];
    for (i = 0, d = 0, l = N_ROWS; i < l; i++) {
        for (j = i + 1; j < l; j++) {
            votes[d] = decisions[d] > 0 ? i : j;
            d = d + 1;
        }
    }

    int amounts[N_CLASSES];
    for (i = 0, l = N_CLASSES; i < l; i++) {
        amounts[i] = 0;
    }
    for (i = 0; i < N_INTERCEPTS; i++) {
        amounts[votes[i]] += 1;
    }

    int classVal = -1;
    int classIdx = -1;
    for (i = 0; i < N_CLASSES; i++) {
        if (amounts[i] > classVal) {
            classVal = amounts[i];
            classIdx= i;
        }
    }
    return classIdx;

}


};
