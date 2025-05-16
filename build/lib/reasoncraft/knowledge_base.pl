% --- 荣格八维认知功能定义 ---
% function(名称, 态度, 缩写).
% 名称: sensing (感觉), intuition (直觉), thinking (思考), feeling (情感)
% 态度: extraverted (外向), introverted (内向)

cognitive_function(sensing,   extraverted, se). % 外向感觉 (Se)
cognitive_function(sensing,   introverted, si). % 内向感觉 (Si)
cognitive_function(intuition, extraverted, ne). % 外向直觉 (Ne)
cognitive_function(intuition, introverted, ni). % 内向直觉 (Ni)
cognitive_function(thinking,  extraverted, te). % 外向思考 (Te)
cognitive_function(thinking,  introverted, ti). % 内向思考 (Ti)
cognitive_function(feeling,   extraverted, fe). % 外向情感 (Fe)
cognitive_function(feeling,   introverted, fi). % 内向情感 (Fi)

% 辅助谓词：检查一个缩写是否是合法的认知功能
is_cognitive_function_abbr(Abbr) :-
    cognitive_function(_, _, Abbr).

% --- 认知功能描述 (可选，用于提供更多信息) ---
function_description(se, "关注当前时刻的感官体验和具体细节 (Focuses on present moment sensory experiences and concrete details)").
function_description(si, "依赖过去的经验和内部存储的感官印象 (Relies on past experiences and internally stored sensory impressions)").
function_description(ne, "探索可能性、模式和外部世界的联系 (Explores possibilities, patterns, and connections in the external world)").
function_description(ni, "洞察深层含义、未来趋势和内在愿景 (Perceives underlying meanings, future trends, and inner visions)").
function_description(te, "通过逻辑和客观标准组织和管理外部世界 (Organizes and manages the external world through logic and objective criteria)").
function_description(ti, "分析信息，寻求精确和内部逻辑一致性 (Analyzes information, seeking precision and internal logical consistency)").
function_description(fe, "寻求和谐，关注他人情感和集体价值观 (Seeks harmony, focuses on others' emotions and collective values)").
function_description(fi, "基于个人价值观和内在信念做决策 (Makes decisions based on personal values and inner convictions)").


% --- 16种MBTI类型及其主要功能栈 ---
% mbti_type(类型名称, 主导功能, 辅助功能, 第三功能, 劣势功能).
% Dominant, Auxiliary, Tertiary, Inferior

% SJ 型 (守护者)
mbti_type(istj, si, te, fi, ne).
mbti_type(isfj, si, fe, ti, ne).
mbti_type(estj, te, si, ne, fi).
mbti_type(esfj, fe, si, ne, ti).

% SP 型 (探险家/工匠)
mbti_type(istp, ti, se, ni, fe).
mbti_type(isfp, fi, se, ni, te).
mbti_type(estp, se, ti, fe, ni).
mbti_type(esfp, se, fi, te, ni).

% NT 型 (理性者)
mbti_type(intj, ni, te, fi, se).
mbti_type(intp, ti, ne, si, fe).
mbti_type(entj, te, ni, se, fi).
mbti_type(entp, ne, ti, fe, si).

% NF 型 (理想主义者)
mbti_type(infj, ni, fe, ti, se).
mbti_type(infp, fi, ne, si, te).
mbti_type(enfj, fe, ni, se, ti).
mbti_type(enfp, ne, fi, te, si).


% --- 功能关系 ---

% 获取一个类型的主导功能
dominant_function(Type, Dominant) :-
    mbti_type(Type, Dominant, _, _, _).

% 获取一个类型的辅助功能
auxiliary_function(Type, Auxiliary) :-
    mbti_type(Type, _, Auxiliary, _, _).

% 获取一个类型的第三功能
tertiary_function(Type, Tertiary) :-
    mbti_type(Type, _, _, Tertiary, _).

% 获取一个类型的劣势功能
inferior_function(Type, Inferior) :-
    mbti_type(Type, _, _, _, Inferior).

% 获取一个类型的前四个功能（主要功能栈）
primary_stack(Type, [Dominant, Auxiliary, Tertiary, Inferior]) :-
    mbti_type(Type, Dominant, Auxiliary, Tertiary, Inferior).

% 查找一个认知功能的相反态度版本
% 例如：se 的相反态度是 si, ni 的相反态度是 ne
opposite_attitude_function(Func1Abbr, Func2Abbr) :-
    cognitive_function(Name, extraverted, Func1Abbr),
    cognitive_function(Name, introverted, Func2Abbr).
opposite_attitude_function(Func1Abbr, Func2Abbr) :-
    cognitive_function(Name, introverted, Func1Abbr),
    cognitive_function(Name, extraverted, Func2Abbr).

% --- 八功能模型 (包括影子功能) ---
% 影子功能通常是主要功能栈中每个功能的相反态度版本
% 5th: Opposing Role (主导功能的相反态度)
% 6th: Critical Parent (辅助功能的相反态度)
% 7th: Trickster (第三功能的相反态度)
% 8th: Demon/Transformative (劣势功能的相反态度)

shadow_functions(Type, [Opposing, Critical, Trickster, Demon]) :-
    mbti_type(Type, Dom, Aux, Ter, Inf),
    opposite_attitude_function(Dom, Opposing),
    opposite_attitude_function(Aux, Critical),
    opposite_attitude_function(Ter, Trickster),
    opposite_attitude_function(Inf, Demon).

% 获取一个类型的完整八功能模型
full_function_stack(Type, FullStack) :-
    primary_stack(Type, Primary),
    shadow_functions(Type, Shadow),
    append(Primary, Shadow, FullStack). % append 是Prolog内置的列表连接谓词

% --- 示例查询 ---
% ?- cognitive_function(Name, Attitude, se).
%    Name = sensing, Attitude = extraverted.
%
% ?- function_description(ni, Desc).
%    Desc = "洞察深层含义、未来趋势和内在愿景 (Perceives underlying meanings, future trends, and inner visions)".
%
% ?- primary_stack(intj, Stack).
%    Stack = [ni, te, fi, se].
%
% ?- shadow_functions(intj, ShadowStack).
%    ShadowStack = [ne, ti, fe, si].
%
% ?- full_function_stack(intj, FullModel).
%    FullModel = [ni, te, fi, se, ne, ti, fe, si].
%
% ?- mbti_type(Type, ni, _, _, _). % 哪些类型的主导功能是Ni?
%    Type = intj ;
%    Type = infj.
%
% ?- opposite_attitude_function(te, Opposite).
%    Opposite = ti.
%
% ?- opposite_attitude_function(Opposite, te).
%    Opposite = ti.