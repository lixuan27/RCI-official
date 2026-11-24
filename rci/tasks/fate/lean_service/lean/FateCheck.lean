namespace RCIFate

open Lean in
def kindTag : ConstantInfo → String
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "def"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quot"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "ctor"
  | .recInfo _ => "rec"

open Lean in
def localClosure (env : Environment) (roots : Array Name) : Array Name := Id.run do
  let mut seen : NameSet := {}
  let mut stack : List Name := roots.toList
  let mut out : Array Name := #[]
  let mut fuel := 100000
  while fuel > 0 do
    fuel := fuel - 1
    match stack with
    | [] => break
    | n :: rest =>
      stack := rest
      if seen.contains n then continue
      seen := seen.insert n
      if (env.getModuleIdxFor? n).isSome then continue
      match env.find? n with
      | none => continue
      | some ci =>
        out := out.push n
        stack := ci.type.getUsedConstants.toList ++ stack
        match ci.value? (allowOpaque := true) with
        | some v => stack := v.getUsedConstants.toList ++ stack
        | none => pure ()
  return out

open Lean in
def jsonStrArr (xs : Array Name) : String :=
  "[" ++ ", ".intercalate (xs.toList.map fun n => (Json.str n.toString).compress) ++ "]"

open Lean Elab Command in
elab "#fate_fingerprint " id:ident nonce:str : command => do
  let env ← getEnv
  let n := id.getId
  let nz := (Json.str nonce.getString).compress
  match env.find? n with
  | none =>
    logInfo m!"FATE_FP \{\"nonce\":{nz},\"name\":{(Json.str n.toString).compress},\"found\":false}"
  | some ci =>
    let deps := (localClosure env ci.type.getUsedConstants).filter (· != n)
    let mut h : UInt64 := mixHash (hash ci.levelParams) ci.type.hash
    let mut anyUnsafe := ci.isUnsafe
    for d in deps.qsort (fun a b => a.toString < b.toString) do
      if let some dc := env.find? d then
        anyUnsafe := anyUnsafe || dc.isUnsafe
        let v := match dc.value? (allowOpaque := true) with
          | some v => v.hash
          | none => 0
        h := mixHash h (mixHash (hash d) (mixHash (hash (kindTag dc))
              (mixHash (hash dc.levelParams) (mixHash dc.type.hash v))))
    let axs ← Lean.collectAxioms n
    let nameJ := (Json.str n.toString).compress
    let kindJ := (Json.str (kindTag ci)).compress
    logInfo m!"FATE_FP \{\"nonce\":{nz},\"name\":{nameJ},\"found\":true,\"kind\":{kindJ},\"fp\":\"{h}\",\"axioms\":{jsonStrArr axs},\"deps\":{jsonStrArr deps},\"unsafe\":{anyUnsafe}}"

end RCIFate
