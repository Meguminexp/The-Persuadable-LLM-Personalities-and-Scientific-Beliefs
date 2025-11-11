# The-Persuadable-LLM-Personalities-and-Scientific-Beliefs

简要说明：该仓库包含一个用于评估与劝导大型语言模型（LLM）的人格特质与科学信念稳健性的实验管线，核心实现位于 [main.py](main.py)。

快速开始
1. 在 dev container 中打开终端。
2. 安装依赖（如果未安装）：  
   pip install -r requirements.txt  # 根据需要创建 requirements.txt（numpy, matplotlib, scipy, openai 等）
3. 配置 API
4. 运行主程序：  
   python main.py

主要配置项（位于 [`main.py`](main.py)）
- 实验温度：[`main.TEMPERATURE`](main.py)  
- 运行次数：[`main.N_RUNS`](main.py)  
- 大五人格定义：[`main.traits`](main.py)  
- 科学信念项：[`main.scientific_belief_items`](main.py)

主要功能（可直接在源码中查看）
- 运行人格实验：[`main.run_trait_experiment`](main.py)  
- 运行单项信念实验：[`main.run_belief_experiment`](main.py)  
- 运行全部信念实验并做汇总：[`main.run_scientific_belief_experiment`](main.py)  
- 生成特质驱动的劝导文本：[`main.get_trait_specific_persuasion`](main.py)  
- 与模型的请求封装：[`main.query_model`](main.py)  
- 构建问卷提示：[`main.build_prompt`](main.py) 与 [`main.build_belief_prompt`](main.py)  
- 保存元数据：[`main.save_experiment_metadata`](main.py)

输出与日志
- 所有原始响应与每次运行的元数据会保存在 `logs/` 目录（由代码自动创建）。  
- 可视化结果会保存为 PNG（例如 `revised_directional_robustness_analysis.png`、`comprehensive_belief_persuasion_analysis.png`、`enhanced_personality_vs_belief_comparison.png`）。

注意事项
- 实验设计：默认 [`main.N_RUNS`](main.py)=25，可根据需要调整以获得更稳定的统计结果。  
- 结果解释：代码包含方向性变换与成功判定逻辑（见 [`main.calculate_directed_changes`](main.py)），阅读源码以了解判定细节。

源码位置
- 主程序：[main.py](main.py)

引用与作者
- This is the final project work for CSC5010 NLP. Thanks for the guidance of Prof. Wang Benyou and all TAs.
- 实验参考：`LLM Personality & Scientific Belief Robustness Experiment v3` （见源码常量 `EXPERIMENT_REFERENCE`）。  
- 作者标注：`AUTHOR = "Research Team"`（见 [`main.py`](main.py)）

如需补充 README 或添加使用示例、依赖清单或 CI 配置，请告知要点。
