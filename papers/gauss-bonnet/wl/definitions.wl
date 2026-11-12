(* ::Package:: *)

(* ------------------------------------------------------------------ *)
(* definitions.wl                                                     *)
(*                                                                    *)
(* (d+1) 维 FLRW 时空与 SGB 模型的定义，供 eoms.wl 加载使用。            *)
(* 空间维数 dimx（即 d）保持符号，不在本文件里代入任何数值。              *)
(*                                                                    *)
(* 度规： ds^2 = -dt^2 + a(t)^2 gamma_ab dx^a dx^b                    *)
(*   gamma_ab 为 dimx 维常曲率空间切片度规，曲率因子 k（k=0 为平直）。    *)
(* ------------------------------------------------------------------ *)

Quiet[Needs["Peanotica`"], Needs::nocont];

(* ---------------- 维数与常数 ---------------- *)
ClearAll[dimx, dimf, k, κ, Λ];
SetAttributes[{dimx, dimf, k, κ, Λ}, Constant];
dimf = dimx + 1;

(* ---------------- 标量函数 ---------------- *)
ClearAll[scaleA, phi, potV, xi, hubbleH];
DefScalarFunction[scaleA, "a"];
DefScalarFunction[phi, "φ"];
DefScalarFunction[potV, "V"];
DefScalarFunction[xi, "ξ"];
DefScalarFunction[hubbleH, "H"];

(* ---------------- 流形 ---------------- *)
(* mf: (d+1) 维全时空；mx: d 维空间切片 *)
mfInfo = SetupRiemannManifold[mf, dimf, gMf, cdMf, {RiemMf, RicMf, RsMf}];
mxInfo = SetupRiemannManifold[mx, dimx, gMx, cdMx, {RiemMx, RicMx, RsMx}];

(* detg = det(g_{ab}) = -a^(2 d)，sqrt(-g) = a^d *)
mfInfo["DetMetric"] = detg;
cdMf[detg, _] = 0;

(* ---------------- 导数 ---------------- *)
ClearAll[pdT];
DefParametreDerivativeOperator[pdT, DerParameters -> {t}];

(* a(t)、t 以及空间切片度规都不随空间坐标变化 *)
pdT[gMx[_, _], _] = 0;
cdMx[scaleA, _] = 0;
cdMx[t, _] = 0;
cdMx[pdT[expr_, ders__], a_] := pdT[cdMx[expr, a], ders];

(* ---------------- FLRW 度规 ---------------- *)
flrwMetric = SparseArray[{
    {1, 1} -> ETensor[-1, {Null, Null}],
    {2, 2} -> ETensor[scaleA^2 gMx[DI@a, DI@b], {a, b}]
}];

flrwMetricInv = SparseArray[{
    {1, 1} -> ETensor[-1, {Null, Null}],
    {2, 2} -> ETensor[scaleA^-2 gMx[a, b], {a, b}]
}];

(* 空间块的协变导数曲率用 k*SymmetricRiemann 表示常曲率切片 *)
flrwPdSpec = {
    {"Pd", pdT[#, t] &},
    {"CovD", cdMx, k*SymmetricRiemann[gMx, #] &}
};

flrwVals = ComputeCurvature[<|
    "Metric" -> flrwMetric,
    "InverseMetric" -> flrwMetricInv,
    "PdSpec" -> flrwPdSpec
|>];

(* frw 是之后 NITensorReduceExtract[expr, frees, frw] 使用的 reducer *)
PopulateCurvatureTReducerRules[frw, mfInfo, flrwVals];
