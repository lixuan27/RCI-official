import Mathlib
open Lean Meta Elab Command

namespace RCIExport

def keepModule (m : Name) : Bool :=
  match m.getRoot with
  | `Mathlib | `Batteries | `Init | `Std => true
  | _ => false

def badSuffixes : List String :=
  ["injEq", "sizeOf_spec", "noConfusion", "noConfusionType", "below", "brecOn",
   "binductionOn", "ibelow", "recOn", "casesOn", "ctorIdx", "toCtorIdx",
   "ctorElim", "ctorElimType", "eq_def", "eq_unfold"]

def isEqnLemma (s : String) : Bool :=
  s.startsWith "eq_" && s.length > 3 && (s.toList.drop 3).all Char.isDigit

def skipName (env : Environment) (n : Name) : Bool :=
  n.isInternal || isPrivateName n || isAuxRecursor env n || isNoConfusion env n ||
  (match n with
   | .str _ s => s.startsWith "match_" || s.startsWith "proof_" || isEqnLemma s ||
                 s.startsWith "_" || badSuffixes.contains s
   | _ => false) ||
  n.getRoot == `Lean || n.getRoot == `Lake

def kindOf : ConstantInfo → String
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "def"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quot"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "ctor"
  | .recInfo _ => "rec"

end RCIExport

open RCIExport in
#eval show CommandElabM Unit from do
  let out := (← IO.getEnv "FATE_EXPORT_OUT").getD "decls.jsonl"
  let shard := ((← IO.getEnv "FATE_EXPORT_SHARD").bind String.toNat?).getD 0
  let nsh := ((← IO.getEnv "FATE_EXPORT_NSHARDS").bind String.toNat?).getD 1
  let env ← getEnv
  let names := env.constants.toList.filterMap fun (n, _) =>
    if (hash n).toNat % nsh == shard then some n else none
  let h ← IO.FS.Handle.mk out .write
  let mut kept := 0
  let mut failed := 0
  let t0 ← IO.monoMsNow
  for n in names do
    let some ci := env.find? n | continue
    let some modIdx := env.getModuleIdxFor? n | continue
    let some modName := env.header.moduleNames[modIdx.toNat]? | continue
    if !keepModule modName then continue
    if skipName env n then continue
    if Meta.isInstanceCore env n then continue
    let sig ← liftTermElabM do
      try
        let f ← withTheReader Core.Context (fun c => { c with maxHeartbeats := 20000 * 1000 }) <|
          PrettyPrinter.ppSignature n
        pure (some (f.fmt.pretty 100))
      catch _ => pure none
    let sig ← match sig with
      | some s => pure s
      | none => do failed := failed + 1; pure s!"{n} : <pp failed>"
    let doc ← findDocString? env n
    let j := Json.mkObj [("name", Json.str n.toString), ("kind", Json.str (kindOf ci)),
      ("module", Json.str modName.toString), ("signature", Json.str sig),
      ("doc", match doc with | some d => Json.str d | none => Json.null)]
    h.putStrLn j.compress
    kept := kept + 1
  h.flush
  let t1 ← IO.monoMsNow
  logInfo m!"EXPORT_DONE shard={shard}/{nsh} candidates={names.length} kept={kept} ppfailed={failed} ms={t1 - t0}"
