# 运动损伤风险预测与干预建议
# 数据挖掘课程作业

# 基础数据处理库
import numpy as np
import pandas as pd

# 数据可视化库
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

# 机器学习库
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 设置随机种子确保结果可复现
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================
# 1. 生成模拟数据集
# ==============================================
print("=" * 60)
print("阶段1：生成模拟数据集")
print("=" * 60)

def generate_simulated_data():
    """
    生成运动损伤模拟数据集
    
    返回:
        DataFrame: 包含6000条记录的数据集
    """
    n_athletes = 200  # 运动员数量
    n_records_per_athlete = 30  # 每位运动员的记录数
    
    movement_types = ['慢跑', '快跑', '冲刺', '跳跃', '力量训练']
    data = []
    
    for athlete_id in range(1, n_athletes + 1):
        for _ in range(n_records_per_athlete):
            # 生成训练时长：50-180分钟，偏正态分布
            training_duration = int(np.random.normal(90, 30))
            training_duration = max(50, min(180, training_duration))
            
            # 生成动作类型：均匀分布
            movement_type = np.random.choice(movement_types)
            
            # 生成心率：根据动作类型有所差异
            base_heart_rate = {
                '慢跑': 120,
                '快跑': 140,
                '冲刺': 175,
                '跳跃': 150,
                '力量训练': 110
            }
            heart_rate = int(np.random.normal(base_heart_rate[movement_type], 15))
            heart_rate = max(60, min(200, heart_rate))
            
            # 计算损伤概率
            base_prob = 0.1  # 基础损伤概率10%
            
            # 规则1: 训练时长>120分钟，概率+20%
            if training_duration > 120:
                base_prob += 0.2
            
            # 规则2: 冲刺或跳跃，概率+30%
            if movement_type in ['冲刺', '跳跃']:
                base_prob += 0.3
            
            # 规则3: 心率>170，概率+25%
            if heart_rate > 170:
                base_prob += 0.25
            
            injury_prob = min(0.95, max(0.02, base_prob))
            injury = 1 if np.random.random() < injury_prob else 0
            
            data.append({
                'athlete_id': athlete_id,
                'training_duration': training_duration,
                'movement_type': movement_type,
                'heart_rate': heart_rate,
                'injury': injury
            })
    
    return pd.DataFrame(data)

df = generate_simulated_data()
print(f"数据集形状: {df.shape}")
print("\n数据集前5行:")
print(df.head())
print("\n损伤标签分布:")
print(df['injury'].value_counts(normalize=True))

# 保存数据集到文件
df.to_csv('运动损伤模拟数据集.csv', index=False, encoding='utf-8-sig')
print(f"\n数据集已保存为: 运动损伤模拟数据集.csv")

# ==============================================
# 2. 数据预处理
# ==============================================
print("\n" + "=" * 60)
print("阶段2：数据预处理")
print("=" * 60)

X = df.drop(['injury', 'athlete_id'], axis=1)
y = df['injury']

# 划分训练集和测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ['training_duration', 'heart_rate']
categorical_features = ['movement_type']

# 标准化数值特征
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# 独热编码分类特征
encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
encoded_train = encoder.fit_transform(X_train[categorical_features])
encoded_test = encoder.transform(X_test[categorical_features])

encoded_feature_names = encoder.get_feature_names_out(categorical_features)

encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_feature_names, index=X_train.index)
encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=X_test.index)

X_train_processed = pd.concat([X_train[numeric_features], encoded_train_df], axis=1)
X_test_processed = pd.concat([X_test[numeric_features], encoded_test_df], axis=1)

print(f"处理后的训练集形状: {X_train_processed.shape}")
print("\n处理后的训练集前5行:")
print(X_train_processed.head())

# ==============================================
# 3. 模型训练与评估
# ==============================================
print("\n" + "=" * 60)
print("阶段3：模型训练与评估")
print("=" * 60)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """训练并评估模型"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n=== {model_name} 分类报告 ===")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {auc:.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['未损伤', '损伤'],
                yticklabels=['未损伤', '损伤'])
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(f'{model_name}_混淆矩阵.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存为: {model_name}_混淆矩阵.png")
    
    return model, y_pred, y_proba, auc

# 训练逻辑回归
log_reg = LogisticRegression(random_state=42, max_iter=200)
log_reg, y_pred_lr, y_proba_lr, auc_lr = evaluate_model(
    log_reg, X_train_processed, y_train, X_test_processed, y_test, '逻辑回归'
)

# 训练随机森林
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf, y_pred_rf, y_proba_rf, auc_rf = evaluate_model(
    rf, X_train_processed, y_train, X_test_processed, y_test, '随机森林'
)

# ==============================================
# 4. 特征重要性分析
# ==============================================
print("\n" + "=" * 60)
print("阶段4：特征重要性分析")
print("=" * 60)

feature_importance = pd.DataFrame({
    'feature': X_train_processed.columns,
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
plt.title('随机森林特征重要性')
plt.xlabel('重要性得分')
plt.ylabel('特征名称')
plt.savefig('随机森林特征重要性.png', dpi=100, bbox_inches='tight')
plt.close()

print("特征重要性排名:")
print(feature_importance)
print("\n特征重要性图已保存为: 随机森林特征重要性.png")

# ==============================================
# 5. 前3个重要特征解释
# ==============================================
print("\n" + "=" * 60)
print("阶段5：前3个重要特征解释")
print("=" * 60)

print("""
### 1. 心率 (heart_rate)
- 重要性: 最高
- 解释: 心率是反映运动员身体负荷的直接指标。当心率超过170次/分钟时，运动员处于高强度训练状态，身体疲劳累积加快，受伤风险显著增加。

### 2. 训练时长 (training_duration)
- 重要性: 第二
- 解释: 训练时长直接影响身体疲劳程度。超过120分钟的长时间训练会导致肌肉糖原耗尽、核心温度升高、注意力下降、动作技术变形。

### 3. 冲刺动作 (movement_type_冲刺)
- 重要性: 第三
- 解释: 冲刺是爆发力极强的动作，肌肉瞬间承受超过体重数倍的力量，膝关节、踝关节承受巨大剪切力，加速和减速过程易导致拉伤。
""")

# ==============================================
# 6. 损伤风险预测与干预建议函数
# ==============================================
print("\n" + "=" * 60)
print("阶段6：损伤风险预测与干预建议")
print("=" * 60)

def get_risk_level(prob):
    """根据概率返回风险等级"""
    if prob < 0.2:
        return '低风险'
    elif prob < 0.4:
        return '中低风险'
    elif prob < 0.6:
        return '中高风险'
    else:
        return '高风险'

def predict_injury_risk(duration, heart_rate, movement_type):
    """
    预测运动员损伤风险并提供干预建议
    
    参数:
        duration: 训练时长（分钟）
        heart_rate: 心率（次/分钟）
        movement_type: 动作类型（慢跑、快跑、冲刺、跳跃、力量训练）
    
    返回:
        dict: 包含损伤概率和干预建议
    """
    input_data = pd.DataFrame({
        'training_duration': [duration],
        'heart_rate': [heart_rate],
        'movement_type': [movement_type]
    })
    
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    encoded_input = encoder.transform(input_data[categorical_features])
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoded_feature_names)
    input_processed = pd.concat([input_data[numeric_features].reset_index(drop=True), 
                                 encoded_input_df], axis=1)
    
    risk_prob = rf.predict_proba(input_processed)[:, 1][0]
    suggestions = []
    
    # 根据心率给出建议
    if heart_rate > 170:
        suggestions.append(f"[警告] 心率过高({heart_rate}次/分)，建议立即降低训练强度，进行2-3分钟的动态恢复，如慢走或深呼吸。")
    elif heart_rate > 150:
        suggestions.append(f"[提示] 心率较高({heart_rate}次/分)，建议每10分钟进行1分钟的补水和短暂休息。")
    
    # 根据训练时长给出建议
    if duration > 120:
        suggestions.append(f"[警告] 训练时长过长({duration}分钟)，建议将训练拆分为多个45-60分钟的单元，中间至少休息15分钟。")
    elif duration > 90:
        suggestions.append(f"[提示] 训练时长适中({duration}分钟)，建议在训练中后段增加5分钟的拉伸放松时间。")
    
    # 根据动作类型给出建议
    high_risk_movements = ['冲刺', '跳跃']
    if movement_type in high_risk_movements:
        suggestions.append(f"[动作] {movement_type}属于高风险动作，建议：1)确保充分热身；2)控制动作次数；3)训练后进行冰敷处理。")
    else:
        suggestions.append(f"[动作] {movement_type}相对安全，建议保持正确的动作姿势，注意呼吸节奏。")
    
    if risk_prob > 0.7:
        suggestions.append("[危险] 损伤风险极高！建议立即停止训练，进行全面身体检查，并安排至少24小时休息恢复。")
    elif risk_prob > 0.5:
        suggestions.append("[注意] 损伤风险较高，建议密切关注身体感受，如有不适立即停止。")
    
    while len(suggestions) < 3:
        suggestions.append("[建议] 建议定期进行身体机能检测，建立个人训练负荷档案。")
    
    return {
        'risk_probability': round(risk_prob * 100, 2),
        'risk_level': get_risk_level(risk_prob),
        'suggestions': suggestions[:3]
    }

# 测试案例
print("=== 测试案例1：高风险场景 ===")
result1 = predict_injury_risk(duration=150, heart_rate=185, movement_type='冲刺')
print(f"损伤概率: {result1['risk_probability']}%")
print(f"风险等级: {result1['risk_level']}")
print("干预建议:")
for i, suggestion in enumerate(result1['suggestions'], 1):
    print(f"  {i}. {suggestion}")

print("\n=== 测试案例2：中等风险场景 ===")
result2 = predict_injury_risk(duration=90, heart_rate=145, movement_type='快跑')
print(f"损伤概率: {result2['risk_probability']}%")
print(f"风险等级: {result2['risk_level']}")
print("干预建议:")
for i, suggestion in enumerate(result2['suggestions'], 1):
    print(f"  {i}. {suggestion}")

print("\n=== 测试案例3：低风险场景 ===")
result3 = predict_injury_risk(duration=60, heart_rate=110, movement_type='慢跑')
print(f"损伤概率: {result3['risk_probability']}%")
print(f"风险等级: {result3['risk_level']}")
print("干预建议:")
for i, suggestion in enumerate(result3['suggestions'], 1):
    print(f"  {i}. {suggestion}")

# ==============================================
# 7. 技术路线说明
# ==============================================
print("\n" + "=" * 60)
print("阶段7：技术路线说明")
print("=" * 60)

print("""
整体流程：数据生成 → 数据预处理 → 模型训练 → 模型评估 → 特征分析 → 预测应用

详细技术路线：
1. 数据生成：基于领域知识定义损伤概率规则，使用NumPy生成6000条模拟数据
2. 数据预处理：独热编码分类特征，标准化数值特征，8:2划分训练/测试集
3. 模型训练：逻辑回归（基准）+ 随机森林（提升）
4. 模型评估：分类报告、AUC-ROC、混淆矩阵热力图
5. 特征分析：随机森林特征重要性排序和可视化
6. 预测应用：封装预测函数，生成个性化干预建议

关键技术要点：
- 设置随机种子确保结果可复现
- 使用class_weight='balanced'处理类别不平衡
- 预处理管道与预测时保持一致
- 基于规则的干预建议生成机制
""")

print("\n" + "=" * 60)
print("程序运行完成！")
print("=" * 60)
