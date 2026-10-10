# 适配器与基础模型

Built with Qwen.

本目录的 clean_e 和 dpo 是 Qwen2.5-VL-3B-Instruct 的 LoRA 修改参数，不是完整模型。E 增加清洗样本后的候选排序训练；DPO 为同一图片对 Yes/No 偏好对照。训练用途是商品匹配，不保证识别真实材质或尺寸。

基础模型按 base_manifest.json 下载。权重和校准文件一起分发，推理代码会校验身份及实现哈希。校准使用验证集、目标 FAR 10%、K=12；不保证新场景或改变 K 后仍具有相同错误接受率。

许可见 ../LICENSES/Qwen-Research.txt 和 ../NOTICE，适配器沿用适用的上游限制。历史 run.json 仅用于来源及校验，不代表历史数据和训练产物全部附带。
