"""Ejektor E1 i OpenFOAM (ESI, simpleFoam, k-ω SST), 2D-rotasjonssymmetrisk kile.

Kjøres på en GitHub Actions-runner (Ubuntu) med OpenFOAM-miljøet lastet.
Én kjøring = én variant og én nettskala, med flere mottrykk etter hverandre
(hvert nytt mottrykk starter fra forrige løsning).

  python of_ejektor.py --kjoring basis --put "0 10 20 30" --nett 1.0 --np 4

Geometri som i CFD_BESTILLING_E1.md, med to forenklinger (står i rapporten):
  - Reduksjonen Ø441 → Ø198,5 er tatt ut. Suget kommer inn gjennom et rett
    Ø198,5-rør 300 mm oppstrøms sugekammeret.
  - Aksen ligger på r = 0,1 mm (slip-vegg) for å unngå kollapsede celler.
Koordinater: x langs aksen i strømretningen, y = radius. Origo i
sugekammerets overkant. Trykk er kinematisk (p/ρ) relativt hydrostatisk.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess

RHO, NU = 1025.0, 1.35e-6
KILE = 5.0          # grader
R_AKSE = 0.1        # mm
X_INN = -300.0
R_SUG = 99.25

BASIS = dict(dn=26.0, dx=0.0, diff_v=4.0, hals_xD=6.5, Qd=450.0)
VARIANTER = {
    "basis": {},
    "V1": dict(dn=23.2),
    "V2": dict(dn=28.0),
    "V3": dict(dx=33.0),
    "V4": dict(dx=-33.0),
    "V5": dict(dx=66.0),
    "V6": dict(diff_v=6.0),
    "V7": dict(hals_xD=5.0),
    "V8": dict(Qd=550.0),
    "V9": dict(avrund=True),
}


def geometri(p):
    dx, rin, rut = p["dx"], p["dn"] / 2, 16.0
    rsp = min(rut, rin + 1.5)
    li = [(X_INN, 36.8), (151, 36.8), (191, 25.7), (281, 25.7), (311, rin), (331, rin)]
    if p.get("avrund"):
        # V9: avrundet overgang konus -> rett parti (skarpt hjørne ga kavitasjon i basis)
        helning = (25.7 - rin) / 30.0
        li[4:5] = [(303, rin + 8 * helning), (308, rin + 0.9), (313, rin + 0.15), (318, rin)]
    lu = [(331, rsp), (328, rut), (311, rut), (281, 31.5), (191, 31.5), (151, 45.0), (X_INN, 45.0)]
    li = [(x + dx if x > X_INN else x, r) for x, r in li]
    lu = [(x + dx if x > X_INN else x, r) for x, r in lu]
    xh = 397.25
    xd = xh + p["hals_xD"] * 66.0
    xe = xd + (48.5 - 33.0) / math.tan(math.radians(p["diff_v"]))
    hus = [(X_INN, R_SUG), (150, R_SUG), (xh, 33.0), (xd, 33.0), (xe, 48.5), (xe + 1000, 48.5)]
    return li, lu, hus, dict(xh=xh, xd=xd, xe=xe, xut=xe + 1000, xtip=331 + dx)


def lag_nett(p, skala, sti):
    import gmsh
    li, lu, hus, m = geometri(p)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    k = 1e-3
    pts = [(X_INN, R_AKSE)] + li + lu + hus + [(m["xut"], R_AKSE)]
    tg = [gmsh.model.geo.addPoint(x * k, r * k, 0) for x, r in pts]
    n = len(tg)
    ln = [gmsh.model.geo.addLine(tg[i], tg[(i + 1) % n]) for i in range(n)]
    i_sug = len(li) + len(lu)          # lanse ytre sist -> hus først
    roller = {0: "drive", i_sug: "suction", n - 2: "outlet", n - 1: "axis"}
    s = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(ln)])
    gmsh.model.geo.rotate([(2, s)], 0, 0, 0, 1, 0, 0, -math.radians(KILE / 2))
    ut = gmsh.model.geo.revolve([(2, s)], 0, 0, 0, 1, 0, 0, math.radians(KILE), numElements=[1], recombine=True)
    gmsh.model.geo.synchronize()
    front, vol, sider = ut[0][1], ut[1][1], [t for d, t in ut[2:]]
    grupper = {"front": [front], "back": [s], "wall": []}
    for i, t in enumerate(sider):
        grupper.setdefault(roller.get(i, "wall"), []).append(t)
    for navn, lst in grupper.items():
        gmsh.model.addPhysicalGroup(2, lst, name=navn)
    gmsh.model.addPhysicalGroup(3, [vol], name="fluid")

    f = gmsh.model.mesh.field
    h_fin, h_grov = 0.6e-3 * skala, 2.5e-3 * skala

    def boks(vin, x0, x1, y1):
        b = f.add("Box")
        for a, v in (("VIn", vin), ("VOut", h_grov), ("XMin", x0), ("XMax", x1), ("YMin", -1), ("YMax", y1),
                     ("ZMin", -1), ("ZMax", 1), ("Thickness", 0.03)):
            f.setNumber(b, a, v)
        return b

    b1 = boks(h_fin, (m["xtip"] - 60) * k, (m["xe"] + 100) * k, 0.055)
    b2 = boks(1.5e-3 * skala, X_INN * k - 0.01, (m["xtip"] - 40) * k, 0.105)
    vegg = [ln[i] for i in range(n) if i not in roller]
    dv = f.add("Distance")
    f.setNumbers(dv, "CurvesList", vegg)
    f.setNumber(dv, "Sampling", 400)
    th = f.add("Threshold")
    for a, v in (("InField", dv), ("SizeMin", 0.25e-3 * skala), ("SizeMax", h_grov),
                 ("DistMin", 0.5e-3 * skala), ("DistMax", 15e-3)):
        f.setNumber(th, a, v)
    mn = f.add("Min")
    f.setNumbers(mn, "FieldsList", [b1, b2, th])
    f.setAsBackgroundMesh(mn)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    gmsh.write(sti)
    _, e, _ = gmsh.model.mesh.getElements(3)
    antall = sum(len(x) for x in e)
    gmsh.finalize()
    return antall, m


# ---------------------------------------------------------------- OpenFOAM-filer
HODE = "FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {kl};\n    object {ob};\n}}\n"


def skriv(sti, kl, ob, kropp):
    os.makedirs(os.path.dirname(sti), exist_ok=True)
    with open(sti, "w") as fh:
        fh.write(HODE.format(kl=kl, ob=ob) + kropp)


def felt(case, navn, kl, dim, intern, bc):
    linjer = "".join(f"    {k}\n    {{\n" + "".join(f"        {a} {b};\n" for a, b in v.items()) + "    }\n"
                     for k, v in bc.items())
    skriv(f"{case}/0/{navn}", kl, navn,
          f"dimensions {dim};\ninternalField uniform {intern};\nboundaryField\n{{\n{linjer}"
          "    front { type wedge; }\n    back { type wedge; }\n}\n")


def sett_opp(case, p, pout_kpa, m, n_iter, np_):
    v_drive = p["Qd"] / 60000 / (math.pi * (0.0368 ** 2 - (R_AKSE * 1e-3) ** 2))
    pk = pout_kpa * 1000 / RHO
    felt(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)", {
        "drive": {"type": "fixedValue", "value": f"uniform ({v_drive:.5f} 0 0)"},
        "suction": {"type": "pressureInletOutletVelocity", "value": "uniform (0 0 0)"},
        "outlet": {"type": "inletOutlet", "inletValue": "uniform (0 0 0)", "value": "uniform (0 0 0)"},
        "wall": {"type": "noSlip"},
        "axis": {"type": "slip"}})
    felt(case, "p", "volScalarField", "[0 2 -2 0 0 0 0]", "0", {
        "drive": {"type": "zeroGradient"},
        "suction": {"type": "totalPressure", "p0": "uniform 0", "value": "uniform 0"},
        "outlet": {"type": "fixedValue", "value": f"uniform {pk:.4f}"},
        "wall": {"type": "zeroGradient"}, "axis": {"type": "zeroGradient"}})
    felt(case, "k", "volScalarField", "[0 2 -2 0 0 0 0]", "1e-4", {
        "drive": {"type": "turbulentIntensityKineticEnergyInlet", "intensity": "0.05", "value": "uniform 1e-4"},
        "suction": {"type": "inletOutlet", "inletValue": "uniform 1e-5", "value": "uniform 1e-5"},
        "outlet": {"type": "inletOutlet", "inletValue": "uniform 1e-4", "value": "uniform 1e-4"},
        "wall": {"type": "kqRWallFunction", "value": "uniform 1e-4"}, "axis": {"type": "zeroGradient"}})
    felt(case, "omega", "volScalarField", "[0 0 -1 0 0 0 0]", "10", {
        "drive": {"type": "turbulentMixingLengthFrequencyInlet", "mixingLength": "0.005", "value": "uniform 10"},
        "suction": {"type": "inletOutlet", "inletValue": "uniform 1", "value": "uniform 1"},
        "outlet": {"type": "inletOutlet", "inletValue": "uniform 10", "value": "uniform 10"},
        "wall": {"type": "omegaWallFunction", "value": "uniform 10"}, "axis": {"type": "zeroGradient"}})
    felt(case, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "0", {
        "drive": {"type": "calculated", "value": "uniform 0"},
        "suction": {"type": "calculated", "value": "uniform 0"},
        "outlet": {"type": "calculated", "value": "uniform 0"},
        "wall": {"type": "nutkWallFunction", "value": "uniform 0"}, "axis": {"type": "calculated", "value": "uniform 0"}})

    skriv(f"{case}/constant/transportProperties", "dictionary", "transportProperties",
          f"transportModel Newtonian;\nnu {NU};\n")
    skriv(f"{case}/constant/turbulenceProperties", "dictionary", "turbulenceProperties",
          "simulationType RAS;\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs on;\n}\n")
    k = 1e-3
    xe, xd = m["xe"] * k, m["xd"] * k
    fo = f"""
    trykk {{ type pressure; libs (fieldFunctionObjects); mode total; rho rhoInf; rhoInf 1; result totalP;
             writeControl writeTime; }}
    strom {{ type surfaceFieldValue; libs (fieldFunctionObjects); regionType patch; name drive;
             operation sum; fields (phi); writeFields false; writeControl timeStep; writeInterval 20; }}
    stromSug {{ $strom; name suction; }}
    stromUt {{ $strom; name outlet; }}
    pDrive {{ type surfaceFieldValue; libs (fieldFunctionObjects); regionType patch; name drive;
              operation areaAverage; fields (p totalP); writeFields false; writeControl timeStep; writeInterval 20; }}
    pDiff {{ type surfaceFieldValue; libs (fieldFunctionObjects); regionType sampledSurface; name diffut;
             sampledSurfaceDict {{ type cuttingPlane; point ({xe:.5f} 0 0); normal (1 0 0); interpolate true; }}
             operation areaAverage; fields (p); writeFields false; writeControl timeStep; writeInterval 20; }}
    pHals {{ $pDiff; name halsinn; sampledSurfaceDict {{ type cuttingPlane; point ({m['xh'] * k:.5f} 0 0); normal (1 0 0); interpolate true; }} }}
    minmax {{ type fieldMinMax; libs (fieldFunctionObjects); fields (p U); location true; mode magnitude;
              writeControl timeStep; writeInterval 20; }}
    linjer {{ type sets; libs (sampling); writeControl writeTime; interpolationScheme cellPoint; setFormat raw;
              fields (U p);
              sets ( diffvegg {{ type uniform; axis distance; start ({xd:.5f} {0.95 * 0.033:.5f} 0);
                                end ({xe:.5f} {0.95 * 0.0485:.5f} 0); nPoints 200; }}
                     halsinn {{ type uniform; axis y; start ({m['xh'] * k:.5f} 0.0002 0); end ({m['xh'] * k:.5f} 0.0329 0); nPoints 120; }}
                     halsmidt {{ type uniform; axis y; start ({(m['xh'] + m['xd']) / 2 * k:.5f} 0.0002 0); end ({(m['xh'] + m['xd']) / 2 * k:.5f} 0.0329 0); nPoints 120; }}
                     diffut {{ type uniform; axis y; start ({xe:.5f} 0.0002 0); end ({xe:.5f} 0.0484 0); nPoints 120; }} ); }}
    yplus {{ type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }}
    residualer {{ type solverInfo; libs (utilityFunctionObjects); fields (U p k omega); writeControl timeStep; writeInterval 20; }}
"""
    skriv(f"{case}/system/controlDict", "dictionary", "controlDict", f"""application simpleFoam;
startFrom latestTime;
startTime 0;
stopAt endTime;
endTime {n_iter};
deltaT 1;
writeControl timeStep;
writeInterval {n_iter};
purgeWrite 1;
writeFormat ascii;
writePrecision 8;
runTimeModifiable true;
functions
{{{fo}}}
""")
    skriv(f"{case}/system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; grad(k) cellLimited Gauss linear 1;
              grad(omega) cellLimited Gauss linear 1; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);
             div(phi,k) bounded Gauss limitedLinear 1; div(phi,omega) bounded Gauss limitedLinear 1;
             div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.5; }
wallDist { method meshWave; }
""")
    skriv(f"{case}/system/fvSolution", "dictionary", "fvSolution", """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
    residualControl { p 2e-6; U 1e-7; "(k|omega)" 1e-6; }
}
relaxationFactors { equations { U 0.8; ".*" 0.6; } fields { p 0.9; } }
""")
    skriv(f"{case}/system/decomposeParDict", "dictionary", "decomposeParDict",
          f"numberOfSubdomains {np_};\nmethod scotch;\n")


def fiks_boundary(case):
    sti = f"{case}/constant/polyMesh/boundary"
    with open(sti) as fh:
        t = fh.read()
    for navn, typ in (("front", "wedge"), ("back", "wedge"), ("wall", "wall")):
        t = re.sub(rf"(\n\s*{navn}\s*\n\s*\{{[^}}]*?type\s+)\w+;", rf"\g<1>{typ};", t)
        t = re.sub(rf"(\n\s*{navn}\s*\n\s*\{{[^}}]*?physicalType\s+)\w+;", rf"\g<1>{typ};", t)
    with open(sti, "w") as fh:
        fh.write(t)


def sh(cmd, case, logg):
    with open(os.path.join(case, logg), "a") as fh:
        r = subprocess.run(cmd, shell=True, cwd=case, stdout=fh, stderr=subprocess.STDOUT, executable="/bin/bash")
    if r.returncode != 0:
        raise RuntimeError(f"{cmd} feilet, se {case}/{logg}")


# ---------------------------------------------------------------- etterbehandling
def siste_linje(mønster):
    filer = sorted(glob.glob(mønster), key=lambda s: float(s.split(os.sep)[-2]) if s.split(os.sep)[-2].replace('.', '').isdigit() else 0)
    if not filer:
        return None
    with open(filer[-1]) as fh:
        rader = [l.split() for l in fh if l.strip() and not l.startswith("#")]
    return rader[-1] if rader else None


def les_sett(case, navn):
    tider = sorted([d for d in os.listdir(f"{case}/postProcessing/linjer")], key=float)
    d = f"{case}/postProcessing/linjer/{tider[-1]}"
    fil = [f for f in os.listdir(d) if f.startswith(navn + "_")]
    rad = []
    with open(os.path.join(d, fil[0])) as fh:
        for l in fh:
            if l.strip() and not l.startswith("#"):
                rad.append([float(t) for t in l.split()])
    return rad


def fra_logg(logg, monster, n=1):
    """Siste verdi (og verdien n linjer før) av et funksjonsobjekt i solverloggen.
    Loggen er sikrere enn .dat-filene, der kolonner mangler før feltet finnes."""
    tall = r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    funn = re.findall(monster + r"\s*=\s*" + tall, logg)
    return [float(v) for v in funn]


def etterbehandle(case, p, pout_kpa, m, celler, loggfil):
    fak = 360.0 / KILE * 60000          # m³/s i kilen -> l/min hel
    with open(os.path.join(case, loggfil)) as fh:
        logg = fh.read()
    qd_l = fra_logg(logg, r"sum\(drive\) of phi")
    qs_l = fra_logg(logg, r"sum\(suction\) of phi")
    qu = fra_logg(logg, r"sum\(outlet\) of phi")[-1] * fak
    qd, qs = -qd_l[-1] * fak, -qs_l[-1] * fak
    # endring i sug over de siste ~500 iterasjonene (25 utskrifter à 20)
    qs_endr = 100 * (qs_l[-1] - qs_l[-26]) / qs_l[-1] if len(qs_l) > 26 else None
    p_drive = fra_logg(logg, r"areaAverage\(drive\) of p")[-1] * RHO / 1000
    pt_drive = fra_logg(logg, r"areaAverage\(drive\) of totalP")[-1] * RHO / 1000
    p_diff = fra_logg(logg, r"areaAverage\(diffut\) of p")[-1] * RHO / 1000
    p_hals = fra_logg(logg, r"areaAverage\(halsinn\) of p")[-1] * RHO / 1000
    mn = re.findall(r"min\(p\) = (\S+) in cell \d+ at location \((\S+) (\S+) \S+\)", logg)
    pmin = None
    if mn:
        v, xx, yy = (float(t) for t in mn[-1])
        pmin = {"p_min_kPa": round(v * RHO / 1000, 1), "x_mm": round(xx * 1000, 1), "r_mm": round(yy * 1000, 1),
                "p_abs_kPa": round(v * RHO / 1000 + 101.325 + RHO * 9.81 * 2.2 / 1000, 1)}
    dv = les_sett(case, "diffvegg")
    tilbake = [r[0] for r in dv if r[1] < -1e-3]
    res = {
        "kjoring": p["navn"], "p_ut_kPa": pout_kpa, "celler": celler,
        "Q_d_lmin": round(qd, 1), "Q_s_lmin": round(qs, 1), "Q_ut_lmin": round(qu, 1),
        "massebalanse_pst": round(100 * (qd + qs - qu) / max(abs(qu), 1e-9), 3),
        "Q_s_endring_siste500_pst": round(qs_endr, 2) if qs_endr is not None else None,
        "p_drivinnlop_kPa": round(p_drive, 2), "pt_drivinnlop_kPa": round(pt_drive, 2),
        "p_diffut_kPa": round(p_diff, 2),
        "p_halsinn_kPa": round(p_hals, 2),
        "p_abs_halsinn_kPa": round(p_hals + 101.325 + RHO * 9.81 * 2.2 / 1000, 1),
        "tilbakestromning_diffusor": bool(tilbake),
        "tilbake_lengde_mm": round(1000 * (max(tilbake) - min(tilbake)), 1) if tilbake else 0.0,
        **({"p_min": pmin} if pmin else {}),
    }
    res["eta"] = round(res["Q_s_lmin"] * pout_kpa / (res["Q_d_lmin"] * (pt_drive - pout_kpa)), 3) \
        if pt_drive > pout_kpa and pout_kpa > 0 else None
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--kjoring", default="basis")
    a.add_argument("--put", default="20")
    a.add_argument("--nett", type=float, default=1.0)
    a.add_argument("--np", type=int, default=4)
    a.add_argument("--iter", type=int, default=3000)
    a.add_argument("--ut", default="ut")
    g = a.parse_args()
    p = dict(BASIS, **VARIANTER[g.kjoring], navn=g.kjoring)
    case = f"run_{g.kjoring}_n{g.nett:g}"
    shutil.rmtree(case, ignore_errors=True)
    os.makedirs(case)
    celler, m = lag_nett(p, g.nett, f"{case}/mesh.msh")
    putliste = [float(t) for t in g.put.split()]
    sett_opp(case, p, putliste[0], m, g.iter, g.np)
    sh("gmshToFoam mesh.msh", case, "log.gmshToFoam")
    fiks_boundary(case)
    sh("checkMesh", case, "log.checkMesh")
    os.makedirs(g.ut, exist_ok=True)
    resultater, slutt = [], 0
    for nr, pout in enumerate(putliste):
        slutt += g.iter
        if nr > 0:
            # nytt mottrykk: endre utløpet i siste tid og regn videre
            sist = sorted([d for d in os.listdir(case) if re.fullmatch(r"\d+", d)], key=int)[-1]
            sti = f"{case}/{sist}/p"
            with open(sti) as fh:
                t = fh.read()
            t = re.sub(r"(outlet\s*\{[^}]*?value\s+uniform\s+)[-\d.e]+", rf"\g<1>{pout * 1000 / RHO:.4f}", t, flags=re.S)
            with open(sti, "w") as fh:
                fh.write(t)
            cd = f"{case}/system/controlDict"
            with open(cd) as fh:
                t = fh.read()
            t = re.sub(r"endTime \d+;", f"endTime {slutt};", t)
            t = re.sub(r"writeInterval \d+;", f"writeInterval {g.iter};", t, count=1)
            with open(cd, "w") as fh:
                fh.write(t)
        for d in glob.glob(f"{case}/processor*"):
            shutil.rmtree(d)
        sh("decomposePar -latestTime -force" if nr else "decomposePar -force", case, "log.decomposePar")
        sh(f"mpirun --oversubscribe -np {g.np} simpleFoam -parallel", case, f"log.simpleFoam_p{pout:g}")
        sh("reconstructPar -latestTime", case, "log.reconstructPar")
        sh("simpleFoam -postProcess -latestTime -func yPlus", case, "log.yPlus")
        r = etterbehandle(case, p, pout, m, celler, f"log.simpleFoam_p{pout:g}")
        r["nettskala"] = g.nett
        resultater.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        # lagre siste felt for figurer
        sist = sorted([d for d in os.listdir(case) if re.fullmatch(r"\d+", d)], key=int)[-1]
        mal = f"{g.ut}/{case}_p{pout:g}"
        os.makedirs(mal, exist_ok=True)
        sh(f"postProcess -func writeCellCentres -time {sist}", case, "log.cc")
        for fnavn in ("U", "p", "C", "k", "yPlus"):
            if os.path.exists(f"{case}/{sist}/{fnavn}"):
                shutil.copy(f"{case}/{sist}/{fnavn}", f"{mal}/{fnavn}")
        shutil.copytree(f"{case}/postProcessing", f"{mal}/postProcessing", dirs_exist_ok=True)
        for lg in glob.glob(f"{case}/log.*"):
            shutil.copy(lg, mal)
    with open(f"{g.ut}/resultat_{g.kjoring}_n{g.nett:g}.json", "w") as fh:
        json.dump(resultater, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
