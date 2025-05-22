% kb.pl - Prolog Knowledge Base

% --- 谓词元数据 (给LLM参考，实际提取时会通过Python读取并处理) ---
% predicate: capital_of/2 - capital_of(City, Country) - City is the capital of Country.
% type: capital_of(atom, atom).
%
% predicate: in_continent/2 - in_continent(Country, Continent) - Country is in Continent.
% type: in_continent(atom, atom).
%
% predicate: population/2 - population(Entity, PopNumber) - Entity has a population of PopNumber.
% type: population(atom, integer).
%
% predicate: president_of/2 - president_of(Person, Country) - Person is the president of Country.
% type: president_of(atom, atom).
%
% predicate: programming_language_paradigm/2 - programming_language_paradigm(Language, Paradigm) - Language follows Paradigm.
% type: programming_language_paradigm(atom, atom).
%
% predicate: developed_by/2 - developed_by(Software, OrganizationOrPerson) - Software was developed by OrganizationOrPerson.
% type: developed_by(atom, atom).
%
% predicate: parent/2 - parent(Parent, Child) - Parent is a parent of Child.
% type: parent(atom, atom).
%
% predicate: male/1 - male(Person) - Person is male.
% type: male(atom).
%
% predicate: female/1 - female(Person) - Person is female.
% type: female(atom).
%
% rule: father/2 - father(F, C) :- parent(F, C), male(F). - F is the father of C.
% rule: mother/2 - mother(M, C) :- parent(M, C), female(M). - M is the mother of C.
% rule: grandparent/2 - grandparent(GP, GC) :- parent(GP, P), parent(P, GC). - GP is a grandparent of GC.
% rule: sibling/2 - sibling(S1, S2) :- parent(P, S1), parent(P, S2), S1 \= S2. - S1 and S2 are siblings.
% rule: ancestor/2 - ancestor(A, D) :- parent(A, D).
% rule: ancestor/2 - ancestor(A, D) :- parent(A, X), ancestor(X, D). - A is an ancestor of D.
% rule: country_with_large_population/1 - country_with_large_population(Country) :- population(Country, Pop), Pop > 100000000. - Country has a population greater than 100 million.
% --- 结束谓词元数据 ---


% --- 地理知识 ---
capital_of(paris, france).
capital_of(berlin, germany).
capital_of(london, uk).
capital_of(washington_dc, usa).
capital_of(beijing, china).
capital_of(tokyo, japan).
capital_of(moscow, russia).
capital_of(ottawa, canada).

in_continent(france, europe).
in_continent(germany, europe).
in_continent(uk, europe).
in_continent(china, asia).
in_continent(japan, asia).
in_continent(usa, north_america).
in_continent(canada, north_america).
in_continent(russia, eurasia). % Russia spans Europe and Asia

population(china, 1400000000).
population(india, 1380000000). % Example, could be outdated
population(usa, 330000000).
population(france, 67000000).
population(tokyo, 37000000). % Population of Tokyo city

% --- 政治知识 ---
president_of(emmanuel_macron, france). % As of my last update, check current
president_of(joe_biden, usa). % As of my last update, check current

% --- 科技知识 ---
programming_language_paradigm(prolog, logic).
programming_language_paradigm(python, multi_paradigm). % (object_oriented, imperative, functional)
programming_language_paradigm(java, object_oriented).
programming_language_paradigm(haskell, functional).
programming_language_paradigm(c, imperative).

developed_by(python, 'Guido van Rossum').
developed_by(swi_prolog, 'Jan Wielemaker').
developed_by(linux_kernel, 'Linus Torvalds').
developed_by(windows, 'Microsoft').

% --- 家庭关系 ---
parent(john, mary).
parent(john, peter).
parent(susan, mary).
parent(susan, peter).
parent(peter, alice).
parent(peter, bob).
parent(jane, john).
parent(david, susan).

male(john).
male(peter).
male(bob).
male(david).

female(susan).
female(mary).
female(alice).
female(jane).

% --- 派生谓词 (规则) ---
father(F, C) :- parent(F, C), male(F).
mother(M, C) :- parent(M, C), female(M).

sibling(S1, S2) :-
    parent(P, S1),
    parent(P, S2),
    S1 \= S2. % Not the same person

grandparent(GP, GC) :-
    parent(GP, P),
    parent(P, GC).

ancestor(A, D) :- parent(A, D).
ancestor(A, D) :-
    parent(A, X),
    ancestor(X, D).

% --- 业务规则示例 ---
% 一个国家人口超过1亿则被认为是人口大国
country_with_large_population(Country) :-
    population(Country, Pop), % Assuming population/2 is for countries
    Pop > 100000000.

% 查找一个城市的所在大洲
city_continent(City, Continent) :-
    capital_of(City, Country),
    in_continent(Country, Continent).

% 查找使用某种编程范式的语言
language_with_paradigm(Paradigm, Language) :-
    programming_language_paradigm(Language, Paradigm).