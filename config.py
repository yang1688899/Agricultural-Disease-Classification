LABEL_NAMES = ["苹果健康","苹果黑星病一般","苹果黑星病严重","苹果灰斑病","苹果雪松锈病一般",
               "苹果雪松锈病严重","樱桃健康","樱桃白粉病一般","樱桃白粉病严重","玉米健康",
               "玉米灰斑病一般","玉米灰斑病严重","玉米锈病一般","玉米锈病严重","玉米叶斑病一般",
               "玉米叶斑病严重","玉米花叶病毒病","葡萄健康","葡萄黑腐病一般","葡萄黑腐病严重",
               "葡萄轮斑病一般","葡萄轮斑病严重","葡萄褐斑病一般","葡萄褐斑病严重","柑桔健康",
               "柑桔黄龙病一般","柑桔黄龙病严重","桃健康","桃疮痂病一般","桃疮痂病严重","辣椒健康",
               "辣椒疮痂病一般","辣椒疮痂病严重","马铃薯健康","马铃薯早疫病一般","马铃薯早疫病严重",
               "马铃薯晚疫病一般","马铃薯晚疫病严重","草莓健康","草莓叶枯病一般","草莓叶枯病严重",
               "番茄健康","番茄白粉病一般","番茄白粉病严重","番茄疮痂病一般","番茄疮痂病严重","番茄早疫病一般",
               "番茄早疫病严重","番茄晚疫病菌一般","番茄晚疫病菌严重","番茄叶霉病一般","番茄叶霉病严重",
               "番茄斑点病一般","番茄斑点病严重","番茄斑枯病一般","番茄斑枯病严重","番茄红蜘蛛损伤一般",
               "番茄红蜘蛛损伤严重","番茄黄化曲叶病毒病一般","番茄黄化曲叶病毒病严重","番茄花叶病毒病"]

INPUT_SIZE = 224
TRAIN_ANNOTATION_FILE = "F:/AgriculturalDisease/ai_challenger_pdr2018_train_annotations_20181021.json"
TRAIN_DIR = "F:/AgriculturalDisease/AgriculturalDisease_trainingset/images/"
VAL_ANNOTATION_FILE = "F:/AgriculturalDisease/ai_challenger_pdr2018_validation_annotations_20181021.json"
VAL_DIR = "F:/AgriculturalDisease/AgriculturalDisease_validationset/images/"
TEST_DIR = "F:/AgriculturalDisease/AgriculturalDisease_testA/images/"

SAVE_MODEL_PATH = "./save/vgg16_224_categorical_crossentropy.model"

BATCH_SIZE = 16