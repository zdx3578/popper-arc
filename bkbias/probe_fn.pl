
% probe_fn.pl —— 比较 hyp 推理的 outpix 与 exs 中的正例
% 用法：
%   swipl -q -s bkbias/probe_fn.pl -- <BK> <HYP> <EXS>
% 或只看某个 pair：
%   swipl -q -s bkbias/probe_fn.pl -- <BK> <HYP> <EXS> p1

:- initialization(main, main).

main :-
    current_prolog_flag(argv, Argv),
    ( Argv = [BK,HYP,EXS] -> PairOpt = none
    ; Argv = [BK,HYP,EXS,Pair] -> PairOpt = some(Pair)
    ; writeln('Usage: swipl -q -s bkbias/probe_fn.pl -- <BK> <HYP> <EXS> [PairId]'),
      halt(1)
    ),
    consult(BK),
    consult(HYP),
    consult(EXS),

    % 收集 exs 中出现过的 pair
    ( setof(P, X^Y^C^pos(outpix(P,X,Y,C)), Pairs0) -> true ; Pairs0 = [] ),
    ( Pairs0 = [] ->
        writeln('[WARN] exs.pl 中没有 pos(outpix(...)).'), halt(0)
    ; true ),

    filter_pairs(PairOpt, Pairs0, Pairs),

    summary(Pairs, 0,0,0, TP, FN, FP),

    denom1 is TP+FP, denom2 is TP+FN,
    safe_div(TP, denom1, Prec),
    safe_div(TP, denom2, Rec),

    format('\n===== SUMMARY =====~n', []),
    format('Precision: ~2f  Recall: ~2f  TP:~d FN:~d FP:~d~n',
           [Prec,Rec,TP,FN,FP]),

    halt(0).

filter_pairs(none, Pairs, Pairs).
filter_pairs(some(Pair), Pairs0, Pairs) :-
    ( member(Pair, Pairs0) ->
        Pairs = [Pair]
    ; format('[WARN] 指定 pair=~w 不在 exs 正例集合中；可选：~w~n', [Pair, Pairs0]),
      halt(0)
    ).

summary([], TPAcc, FNAcc, FPAcc, TPAcc, FNAcc, FPAcc).
summary([P|Ps], TPAcc, FNAcc, FPAcc, TP, FN, FP) :-
    gold_set(P, Gold),              % 期望
    pred_set(P, Pred),              % 预测
    intersect(Gold, Pred, TPset),
    subtract_set(Gold, Pred, FNset),
    subtract_set(Pred, Gold, FPset),

    length(Gold, GoldN), length(Pred, PredN),
    length(TPset, TPn), length(FNset, FNn), length(FPset, FPn),

    format('\n=== Pair ~w ===~n', [P]),
    format('Gold: ~d  Pred: ~d  TP=~d  FN=~d  FP=~d~n',
           [GoldN, PredN, TPn, FNn, FPn]),
    ( FNn > 0 -> format('FN (missing): ~w~n', [FNset]) ; true ),
    ( FPn > 0 -> format('FP (extra):   ~w~n', [FPset]) ; true ),

    TPAcc2 is TPAcc + TPn, FNAcc2 is FNAcc + FNn, FPAcc2 is FPAcc + FPn,
    summary(Ps, TPAcc2, FNAcc2, FPAcc2, TP, FN, FP).

gold_set(P, Gold) :-
    ( setof((X,Y,C), pos(outpix(P,X,Y,C)), S) -> Gold = S ; Gold = [] ).

pred_set(P, Pred) :-
    findall((X,Y,C), outpix(P,X,Y,C), L),
    sort(L, Pred).

intersect(A,B,Out) :- findall(X, (member(X,A), member(X,B)), T), sort(T,Out).
subtract_set(A,B,Out) :- findall(X, (member(X,A), \+ member(X,B)), T), sort(T,Out).

safe_div(_N, 0, 1.0) :- !.   % 分母为 0 时返回 1.0（避免异常）
safe_div(N, D, R) :- R is N / D.
