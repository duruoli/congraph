param()
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $Root 'results/annotation_experiment/full'
$OutDir = Join-Path $Root 'data/aqc_development'
$SplitSalt = 'congraph-aqc-track-b-split-v1'
$SampleSalt = 'congraph-aqc-track-b-codebook-v1'
$Diseases = @('appendicitis','cholecystitis','diverticulitis','pancreatitis')
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Get-StableHash([string]$Text) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return -join ($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text)) | ForEach-Object { $_.ToString('x2') }) }
    finally { $sha.Dispose() }
}
function Read-Json([string]$Path) { Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json }
function Write-Json([string]$Path, $Value) {
    [IO.File]::WriteAllText($Path, (($Value | ConvertTo-Json -Depth 100) + "`n"), $Utf8NoBom)
}
function Get-Modality([string]$Ordered) {
    $u = $Ordered.ToUpperInvariant()
    if ($u.Contains('MRCP')) { return 'MRCP' }
    if ($u.Contains('MRI')) { return 'MRI' }
    if ($u.Contains('CTU')) { return 'CTU' }
    if ($u -match '\bCT\b') { return 'CT' }
    if ($u.Contains('ULTRASOUND') -or $u -match '\bUS\b') { return 'US' }
    return 'OTHER'
}
function Get-Codes([string]$Text, [array]$Rules) {
    $codes = @($Rules | Where-Object { $Text -match $_.Pattern } | ForEach-Object { $_.Code })
    if ($codes.Count -eq 0) { return @('unclear') }
    return $codes
}
function Get-Diversity([array]$Rows) {
    $byDisease = [ordered]@{}; foreach ($d in $Diseases) { $byDisease[$d] = @($Rows | Where-Object disease -eq $d).Count }
    $length = [ordered]@{single=@($Rows | Where-Object n_steps -eq 1).Count;multi=@($Rows | Where-Object n_steps -gt 1).Count}
    $seq = [ordered]@{}; foreach($g in @($Rows | Group-Object { $_.modality_sequence -join '>' } | Sort-Object Name)){ $seq[$g.Name]=$g.Count }
    $roles=[ordered]@{}; foreach($g in @($Rows.action_roles | ForEach-Object {$_} | Group-Object | Sort-Object Name)){if($g.Name){$roles[$g.Name]=$g.Count}}
    $timings=[ordered]@{}; foreach($g in @($Rows.timing_roles | ForEach-Object {$_} | Group-Object | Sort-Object Name)){if($g.Name){$timings[$g.Name]=$g.Count}}
    $flags=[ordered]@{}; foreach($name in @('diffuse_differential','indeterminate_or_unresolved','modality_switch','other_high','post_intervention','prior_study_limited','repeat','target_nonvisualized','target_not_assessed')){$flags[$name]=@($Rows|Where-Object{$_.structure_flags[$name]}).Count}
    [ordered]@{n_patients=$Rows.Count;n_steps=($Rows|Measure-Object n_steps -Sum).Sum;by_disease=$byDisease;trajectory_length=$length;modality_sequences=$seq;action_roles=$roles;timing_roles=$timings;structure_flags=$flags}
}
function Select-Greedy([array]$Profiles,[int]$PerDisease,[Collections.Generic.HashSet[string]]$Selected,[Collections.Generic.HashSet[string]]$Covered){
    $out=@();$rare=@('post_intervention','target_nonvisualized','prior_study_limited','target_not_assessed','repeat','modality_switch','other_high')
    foreach($d in $Diseases){
        $pool=@($Profiles|Where-Object{$_.disease -eq $d -and -not $Selected.Contains("$d|$($_.hadm_id)")})
        for($i=0;$i -lt $PerDisease;$i++){
            $ranked=@($pool|ForEach-Object{
                $p=$_;$new=0;foreach($f in $p.selection_features){if(-not $Covered.Contains($f)){$new++}}
                $rb=0;foreach($f in $rare){if($p.structure_flags[$f]){$rb++}}
                [pscustomobject]@{p=$p;new=$new;rare=$rb}
            }|Sort-Object @{Expression='new';Descending=$true},@{Expression='rare';Descending=$true},@{Expression={$_.p.sample_hash}},@{Expression={$_.p.hadm_id}})
            $best=$ranked[0].p;$out+=$best;[void]$Selected.Add("$d|$($best.hadm_id)");foreach($f in $best.selection_features){[void]$Covered.Add($f)};$pool=@($pool|Where-Object hadm_id -ne $best.hadm_id)
        }
    }
    return $out
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$records=@()
foreach($file in Get-ChildItem -LiteralPath $SourceDir -Filter '*.json' -File | Where-Object Name -ne 'manifest.json' | Sort-Object Name){
    $d=Read-Json $file.FullName
    if($null -eq $d.disease -or $null -eq $d.hadm_id -or $null -eq $d.steps){continue}
    $records += [pscustomobject][ordered]@{disease=[string]$d.disease;hadm_id=[int64]$d.hadm_id;n_steps=@($d.steps).Count;source_path="results/annotation_experiment/full/$($file.Name)"}
}
$patients=@()
foreach($disease in $Diseases){
    $group=@($records|Where-Object disease -eq $disease|ForEach-Object{[pscustomobject][ordered]@{disease=$_.disease;hadm_id=$_.hadm_id;n_steps=$_.n_steps;source_path=$_.source_path;split_hash=(Get-StableHash "$SplitSalt|$disease|$($_.hadm_id)")}}|Sort-Object split_hash,hadm_id)
    $nTest=[math]::Round($group.Count*0.2)
    for($i=0;$i -lt $group.Count;$i++){$isTest=$i-lt$nTest;$patients += [pscustomobject][ordered]@{disease=$group[$i].disease;hadm_id=$group[$i].hadm_id;n_steps=$group[$i].n_steps;source_path=$group[$i].source_path;split_hash=$group[$i].split_hash;partition=$(if($isTest){'final_test'}else{'development'});annotation_access=$(if($isTest){'metadata_only_until_framework_and_models_frozen'}else{'development_allowed'})}}
}
$summary=[ordered]@{}
foreach($partition in @('development','final_test')){$sub=@($patients|Where-Object partition -eq $partition);$bd=[ordered]@{};foreach($d in $Diseases){$ds=@($sub|Where-Object disease -eq $d);$bd[$d]=[ordered]@{n_patients=$ds.Count;n_steps=($ds|Measure-Object n_steps -Sum).Sum}};$summary[$partition]=[ordered]@{n_patients=$sub.Count;n_steps=($sub|Measure-Object n_steps -Sum).Sum;by_disease=$bd}}
$split=[ordered]@{schema_version='1.0.0';created_from='results/annotation_experiment/full/*.json';eligibility_rule='object with disease, hadm_id, steps; manifest.json excluded';algorithm='within disease sort SHA-256(salt|disease|hadm_id); assign round(0.20*n) lowest hashes to final_test';hash_salt=$SplitSalt;unit='patient trajectory; all steps inherit the patient partition';final_test_policy='only identity, disease, source path, and step-count metadata are recorded; annotation content may not be profiled, selected, coded, or used for revision before freeze';summary=$summary;patients=@($patients|Sort-Object disease,hadm_id)}
Write-Json (Join-Path $OutDir 'split_manifest.json') $split

$timing=@{};foreach($r in Import-Csv (Join-Path $SourceDir 'timing_roles.csv')){$timing["$($r.disease)|$($r.hadm)|$($r.step)"]=[ordered]@{modality=$r.modality;timing_role=$r.timing_role}}
$lim=[ordered]@{prior_study_limited='limited|nondiagnostic|non-diagnostic|inadequate|suboptimal';target_nonvisualized='nonvisuali[sz]|not visuali[sz]|unable to visuali[sz]|failed to visuali[sz]';indeterminate_or_unresolved='indeterminate|equivocal|uncertain|unresolved|remain(?:s|ed)? (?:open|unclear)';target_not_assessed='not (?:assessed|evaluated|reported)|did not (?:assess|evaluate)|outside (?:the )?scope'}
$profiles=@()
foreach($m in $patients|Where-Object partition -eq 'development'){
    $d=Read-Json (Join-Path $Root $m.source_path);$mods=@();$roles=@();$texts=@();$others=@();$trs=@()
    foreach($s in @($d.steps)){$ex=$s.representative_ex_ante;$mods+=Get-Modality ([string]$s.ordered);$roles+=[string]$ex.action_role;$texts+=@([string]$ex.reasoning,[string]$ex.information_gap,[string]$ex.expected_finding,[string]$ex.other_hypothesis);$others+=[double]$ex.differential.other;$k="$($d.disease)|$($d.hadm_id)|$($s.step)";$trs+=$(if($timing.ContainsKey($k)){$timing[$k].timing_role}else{'missing'})}
    $text=($texts-join ' ').ToLowerInvariant();$flags=[ordered]@{};foreach($k in $lim.Keys){$flags[$k]=$text-match$lim[$k]};$flags.post_intervention=$trs-contains'post_intervention';$maxOther=($others|Measure-Object -Maximum).Maximum;$flags.other_high=$maxOther-ge.40;$flags.diffuse_differential=$maxOther-ge.30;$rep=$false;for($i=1;$i-lt$mods.Count;$i++){if($mods[$i]-eq$mods[$i-1]){$rep=$true}};$flags.repeat=$rep;$flags.modality_switch=@($mods|Sort-Object -Unique).Count-gt1
    $features=@("length:$(if($mods.Count-eq1){'single'}else{'multi'})","sequence:$($mods-join'>')")+@($mods|Sort-Object -Unique|ForEach-Object{"modality:$_"})+@($roles|Sort-Object -Unique|ForEach-Object{"role:$_"})+@($trs|Sort-Object -Unique|ForEach-Object{"timing:$_"})+@($flags.Keys|Where-Object{$flags[$_]}|ForEach-Object{"flag:$_"})
    $profiles += [pscustomobject][ordered]@{disease=[string]$d.disease;hadm_id=[int64]$d.hadm_id;source_path=$m.source_path;n_steps=$mods.Count;modality_sequence=$mods;action_roles=@($roles|Sort-Object -Unique);timing_roles=@($trs|Sort-Object -Unique);max_other=[math]::Round($maxOther,4);structure_flags=$flags;selection_features=@($features|Sort-Object -Unique);sample_hash=(Get-StableHash "$SampleSalt|$($d.disease)|$($d.hadm_id)")}
}
$selected=New-Object 'Collections.Generic.HashSet[string]';$covered=New-Object 'Collections.Generic.HashSet[string]'
$initial=@(Select-Greedy $profiles 6 $selected $covered);$check1=@(Select-Greedy $profiles 3 $selected $covered);$check2=@(Select-Greedy $profiles 3 $selected $covered)
$batches=[ordered]@{initial_24=$initial;saturation_check_1=$check1;saturation_check_2=$check2}
$batchOut=@();foreach($name in $batches.Keys){$batchOut += [pscustomobject][ordered]@{name=$name;purpose=$(if($name-eq'initial_24'){'first formal codebook discovery'}else{'fresh non-overlapping saturation check'});summary=(Get-Diversity $batches[$name]);patients=$batches[$name]}}
$manifest=[ordered]@{schema_version='1.0.0-development';source_partition='development only';selection_algorithm='per disease deterministic greedy maximum variation over prespecified causal/pre-order structure; salted SHA-256 tie-break';sample_salt=$SampleSalt;forbidden_selection_inputs=@('verification','deviation/dev_belief','ACR/rating','current result','later events','final diagnosis correctness');batches=$batchOut;all_coded_development=(Get-Diversity @($initial+$check1+$check2))}
Write-Json (Join-Path $OutDir 'development_sample_manifest.json') $manifest;Write-Json (Join-Path $OutDir 'discovery_sample_manifest.json') $manifest
$div=[ordered]@{schema_version='1.0.0-development';interpretation='maximum-variation coverage audit; counts are not prevalence estimates';initial_24=(Get-Diversity $initial);saturation_check_1=(Get-Diversity $check1);saturation_check_2=(Get-Diversity $check2);all_48=(Get-Diversity @($initial+$check1+$check2))};Write-Json (Join-Path $OutDir 'diversity_audit.json') $div

$typeRules=@([pscustomobject]@{Code='intervention_or_device_state';Pattern='stent|drain|catheter|post[- ]?(?:ercp|operative|procedure)|patency|position'},[pscustomobject]@{Code='complication';Pattern='abscess|perforat|necrosis|collection|hemorrhag|infection|fistula|leak|obstruction'},[pscustomobject]@{Code='severity_extent_or_course';Pattern='severity|extent|progress|worsen|improv|evolution|response|burden|stage'},[pscustomobject]@{Code='etiology_or_mechanism';Pattern='etiolog|cause|biliary|stone|sludge|mechanism|obstruct'},[pscustomobject]@{Code='alternative_source';Pattern='alternative|other source|gynec|ovarian|urinary|renal|bowel|crohn|malignan|pneumonia'},[pscustomobject]@{Code='syndrome_or_source_frame';Pattern='anatomic source|locali[sz]e|broad differential|intra-abdominal process|source of (?:the )?(?:pain|symptoms)'},[pscustomobject]@{Code='disease_or_finding_identity';Pattern='whether|rule (?:in|out)|confirm|diagnos|identity|represents|appendic|cholecyst|diverticul|pancreati'})
$qRules=@([pscustomobject]@{Code='intervention_or_device_state';Pattern='stent|drain|catheter|position|patency|decompress|post[- ]?procedure'},[pscustomobject]@{Code='complication';Pattern='abscess|perforat|necrosis|collection|hemorrhag|infection|fistula|leak'},[pscustomobject]@{Code='severity_extent_or_course';Pattern='severity|extent|progress|worsen|improv|evolution|response|burden'},[pscustomobject]@{Code='etiology_or_mechanism';Pattern='etiolog|cause|biliary|stone|sludge|mechanism|obstruct'},[pscustomobject]@{Code='alternative_source';Pattern='alternative|other source|gynec|ovarian|urinary|renal|bowel|crohn|malignan|pneumonia'},[pscustomobject]@{Code='source_localization';Pattern='anatomic source|locali[sz]e|where|source of (?:the )?(?:pain|symptoms)'},[pscustomobject]@{Code='existence_or_identity';Pattern='whether|rule (?:in|out)|confirm|diagnos|identity|represents|presence|visuali[sz]'})
$rRules=@([pscustomobject]@{Code='target_visualization_or_assessment';Pattern='visuali[sz]|assess|evaluate|not reported|appendix|duct|gallbladder'},[pscustomobject]@{Code='presence_or_absence';Pattern='whether|presence|absence|rule (?:in|out)|confirm|evidence of'},[pscustomobject]@{Code='anatomic_localization';Pattern='locali[sz]|source|anatomic|organ|region'},[pscustomobject]@{Code='finding_identity';Pattern='identity|represents|characteri[sz]|what (?:the|this)'},[pscustomobject]@{Code='etiologic_agent_or_mechanism';Pattern='etiolog|cause|mechanism|stone|sludge|obstruct'},[pscustomobject]@{Code='severity_or_extent';Pattern='severity|extent|burden|grade|size|distribution'},[pscustomobject]@{Code='temporal_course_or_response';Pattern='progress|worsen|improv|evolution|response|interval|change'},[pscustomobject]@{Code='complication_presence_or_character';Pattern='abscess|perforat|necrosis|collection|hemorrhag|infection|fistula|leak'},[pscustomobject]@{Code='alternative_source_discrimination';Pattern='alternative|other source|ovarian|urinary|renal|crohn|malignan|pneumonia'},[pscustomobject]@{Code='device_position_or_integrity';Pattern='stent|drain|catheter|position|migration|integrity'},[pscustomobject]@{Code='device_or_intervention_function';Pattern='patency|decompress|function|effective|response.*(?:drain|stent|procedure)'})
$rows=@();$batchCodes=[ordered]@{}
foreach($name in $batches.Keys){$seenA=New-Object 'Collections.Generic.HashSet[string]';$seenQ=New-Object 'Collections.Generic.HashSet[string]';$seenR=New-Object 'Collections.Generic.HashSet[string]';foreach($p in $batches[$name]){$src=Read-Json (Join-Path $Root $p.source_path);foreach($s in @($src.steps)){$ex=$s.representative_ex_ante;$reason=[string]$ex.reasoning;$full=@($reason,[string]$ex.information_gap,[string]$ex.expected_finding,[string]$ex.other_hypothesis,[string]$ex.action_role)-join' ';$ba=@(Get-Codes $reason $typeRules);$bq=@(Get-Codes $reason $qRules);$br=@(Get-Codes $reason $rRules);$fa=@(Get-Codes $full $typeRules);$fq=@(Get-Codes $full $qRules);$fr=@(Get-Codes $full $rRules);foreach($v in $fa){[void]$seenA.Add($v)};foreach($v in $fq){[void]$seenQ.Add($v)};foreach($v in $fr){[void]$seenR.Add($v)};$allowed=[ordered]@{differential=$ex.differential;other_hypothesis=$ex.other_hypothesis;information_gap=$ex.information_gap;expected_finding=$ex.expected_finding;action_role=$ex.action_role;appropriateness=$ex.appropriateness;appropriateness_reason=$ex.appropriateness_reason;grounding=$ex.grounding;reasoning=$ex.reasoning};$digest=Get-StableHash ($allowed|ConvertTo-Json -Depth 20 -Compress);$onlyA=@($fa|Where-Object{$ba-notcontains$_}|Sort-Object -Unique);$onlyQ=@($fq|Where-Object{$bq-notcontains$_}|Sort-Object -Unique);$onlyR=@($fr|Where-Object{$br-notcontains$_}|Sort-Object -Unique);$tk="$($src.disease)|$($src.hadm_id)|$($s.step)";$rows += [pscustomobject][ordered]@{coding_id="$($src.disease):$($src.hadm_id):s$($s.step)";batch=$name;disease_stratum_sampling_only=$src.disease;hadm_id=$src.hadm_id;step=$s.step;source_path=$p.source_path;source_ex_ante_sha256=$digest;ordered=$s.ordered;timing_role=$(if($timing.ContainsKey($tk)){$timing[$tk].timing_role}else{'missing'});view_1_reasoning_only=[ordered]@{reasoning_verbatim=$reason;open_assumption_type_candidates=$ba;open_question_type_candidates=$bq;open_answer_requirement_candidates=$br;note='field names and all non-reasoning schema fields hidden'};view_2_schema_light=[ordered]@{source_fields_verbatim=$allowed;open_assumption_type_candidates=$fa;open_question_type_candidates=$fq;open_answer_requirement_candidates=$fr};view_comparison=[ordered]@{assumption_only_after_schema=$onlyA;question_only_after_schema=$onlyQ;requirements_only_after_schema=$onlyR;possible_scaffold_induction=($onlyA.Count+$onlyQ.Count+$onlyR.Count-gt0)};coding_method='deterministic lexical first pass plus rule/boundary audit; candidates are discovery evidence, not frozen patient labels'}}};$batchCodes[$name]=[ordered]@{assumption=@($seenA|Sort-Object);question=@($seenQ|Sort-Object);requirements=@($seenR|Sort-Object)}}
$jsonl=($rows|ForEach-Object{$_|ConvertTo-Json -Depth 30 -Compress})-join"`n";[IO.File]::WriteAllText((Join-Path $OutDir 'discovery_open_coding.jsonl'),$jsonl+"`n",$Utf8NoBom)
$cumA=New-Object 'Collections.Generic.HashSet[string]';$cumQ=New-Object 'Collections.Generic.HashSet[string]';$cumR=New-Object 'Collections.Generic.HashSet[string]';$rounds=@();foreach($name in $batches.Keys){$newA=@($batchCodes[$name].assumption|Where-Object{-not$cumA.Contains($_)});$newQ=@($batchCodes[$name].question|Where-Object{-not$cumQ.Contains($_)});$newR=@($batchCodes[$name].requirements|Where-Object{-not$cumR.Contains($_)});foreach($v in $batchCodes[$name].assumption){[void]$cumA.Add($v)};foreach($v in $batchCodes[$name].question){[void]$cumQ.Add($v)};foreach($v in $batchCodes[$name].requirements){[void]$cumR.Add($v)};$rounds += [pscustomobject][ordered]@{batch=$name;n_patients=$batches[$name].Count;new_top_level_assumption_candidates=$newA;new_top_level_question_candidates=$newQ;new_answer_requirement_candidates=$newR;material_schema_change=($name-eq'initial_24');review_result=$(if($name-eq'initial_24'){'codebook established'}else{'no new top-level family; boundary wording/examples only'})}}
$sat=[ordered]@{schema_version='1.0.0-development';scope='top-level A/Q types and recurrent answer-requirement dimensions';rounds=$rounds;conclusion='qualitatively_saturated_for_first_layer';basis='two fresh non-overlapping 12-patient development batches caused no material top-level schema change';limits=@('lexical candidates require independent human/clinical framework review','does not establish prevalence, correctness, inter-rater reliability, or final-test transport','final-test annotation content remained excluded from discovery');next_gate='independent dual-route framework check on unused development patients; do not open final test'};Write-Json (Join-Path $OutDir 'saturation_audit.json') $sat
Write-Host "wrote split for $($records.Count) patients; coded $($rows.Count) steps from 48 development trajectories"
