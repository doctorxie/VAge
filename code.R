
# 设置文件夹位置
setwd("./")

### 载入R包
library(h2o)
library(ggplot2)
library(shapviz)

# 启动H2O包
h2o.init()

# 加载数据
df <- readRDS("./Web_data_df.Rdata")

data <- as.h2o(df) 
train <- data[data$location_n %in% c("Sanshan","Shapu","Donghan"),]

## 
predictors <- c("BaPWV","ALP","Urea","HbA1c","ALB",
                "Waist2hip_ratio","Heart_rate","Occupation",
                "Sex","Smoking_status","Pulse_pressure",
                "MCV","Sleep_hours","HGB","PLT")
response <- "Age" 

### 最终模型的训练
set.seed(12)

model <- h2o.deeplearning(
  x = predictors,          # 特征列名
  y = response,            # 目标变量列名
  training_frame = train,
  activation = "Rectifier",   # 激活函数
  distribution="gaussian",
  l1=1e-05,
  l2=1e-05,
  # 必须设置的SHAP相关参数
  export_weights_and_biases = TRUE,  # 启用权重导出
  reproducible = TRUE,               # 确保可复现性
  checkpoint = NULL,                  # 禁用检查点
  seed = 12
)


# 第一次运行考虑到R版本可能与我电脑的不同，所以要先建模获得model
# 后续则可跳过建模，直接读入已建好的模型model，提高速度
save.image("./model") # 保存model
load("model") # 读取 model

# 计算SHAP值
background <- data[sort(sample(1:nrow(data), 500)), ]


### 
# 前台输入数据赋值给左侧, 举例如下
Sex = "Male" # 或者 "Female"
Smoking_status = "Current" # 或者 "Former" 或者 "Current"
Occupation = "Worker"#  或者 "Farmer_unemployed","Worker" 或者 "Service_staff" 或者 "White_collar" 或者 "Others"
Sleep_hours = 5
Pulse_pressure = 65
Heart_rate = 70
Waist2hip_ratio = 0.726
BaPWV = 1500
HbA1c = 7.0
ALP = 61
ALB = 45
Urea = 4.86
MCV = 91.2
PLT = 152
HGB = 105

# 将输入的数据整理成数据框
input <- data.frame(Sex = "Male", # 或者 "Female"
                    Smoking_status = "Current", # 或者 "Former" 或者 "Current"
                    Occupation = "Worker", #  或者 "Farmer_unemployed","Worker" 或者 "Service_staff" 或者 "White_collar" 或者 "Others"
                    Sleep_hours = 5,
                    Pulse_pressure = 65,
                    Heart_rate = 70,
                    Waist2hip_ratio = 0.726,
                    BaPWV = 1500,
                    HbA1c = 7.0,
                    ALP = 61,
                    ALB = 45,
                    Urea = 4.86,
                    MCV = 91.2,
                    PLT = 152,
                    HGB = 105)

# 预测
pred <- h2o.predict(model, as.h2o(input))
pred 
# 这个数值就是预测的outoput方框中展示的数值
output <- round(pred,0) # 四舍五入，取整数
output


### 可视化
# 计算SHAP值
shp <- shapviz(model,
               X_pred = input, 
               background=background # 关键参
)

#  出图
piture <- sv_waterfall(shp,
                       row_id = 1,
                       max_display = 15,
                       #fill_colors = 1:2,
                       annotation_size = 6) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    plot.subtitle = element_text(hjust = 0.5, color = "black", size = 12),
    axis.text.x = element_text(size = 10, angle = 0, vjust = 0.5),
    axis.text.y = element_text(size = 10),
    axis.title.x = element_text(margin = margin(t = 10), size = 12),
    # panel.grid = element_blank(),
    plot.caption = element_text(color = "black", hjust = 0, margin = margin(t = 10))# ,
    # plot.margin = margin(1, 1, 1, 1, "cm")
  )
piture
# 将图片对象piture展示在网站中
# 也可先通过ggsave保存然后在展示出来
ggsave("./temp_SHAP_waterfall.pdf",piture,width = 16,height = 10,units = c("cm"))


# 关闭H2O集群
# h2o.shutdown(prompt = FALSE)

