/**********************************************************************
* Prolog Knowledge Base (示例节选)
* 说明:
*   1. 每个谓词都有元数据注释, 供 Python 侧抽取后传给 LLM
*   2. 中文 / 英文别名、类型、示例、NL↔Prolog 对照统一写成注释
**********************************************************************/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  动物学  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% predicate: animal/1
% arity: 1
% type: species
% zh_alias: 动物
% example_nl: "列出所有动物"
% example_pl: animal(X).
animal(lion).
animal(tiger).
animal(cow).
animal(eagle).

% predicate: carnivore/1
% zh_alias: 食肉动物
% rule_desc: 如果某动物以肉为食, 则为食肉动物
% example_nl: "哪些动物是食肉动物?"
% example_pl: carnivore(X).
carnivore(lion).
carnivore(tiger).
carnivore(eagle).

% 组合派生谓词
% predicate: mammal/1
% rule_desc: 哺乳动物 = 动物 ∧ 非鸟类
mammal(X) :- animal(X), \+ bird(X).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  鸟类  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

bird(eagle).
bird(penguin).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  地理  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% predicate: country/1
% zh_alias: 国家
country(usa).
country(france).
country(china).

% predicate: capital/2
% zh_alias: 首都
% 参数1: Country (atom)
% 参数2: City    (atom)
capital(usa, 'Washington D.C.').
capital(france, 'Paris').
capital(china, 'Beijing').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  数论  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% predicate: prime/1
% zh_alias: 质数
prime(2).  prime(3).  prime(5).  prime(7).  prime(11).

% 派生谓词: even/1
% zh_alias: 偶数
even(N) :- 0 is N mod 2.

% example_nl: "哪些质数小于10?"
% example_pl: (prime(X), X<10).