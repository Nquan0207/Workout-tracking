from typing import Dict, List, Optional

# Simple rule engine operating on angle metrics per frame
# Rule format examples:
# {"id":"depth_ok","metric":"knee_angle_min","op":"<=","value":90}
# {"id":"back_neutral","metric":"trunk_angle","range":[10,45]}

def eval_rule(rule:Dict, metrics:Dict[str,float], context: Optional[Dict] = None) -> bool:
    mid = rule['id']
    if context is None:
        context = {}
    states = rule.get('states')
    if states:
        state = context.get('state')
        if state not in states:
            # Rule is not applicable in this movement phase.
            return True
    if 'range' in rule:
        lo, hi = rule['range']
        v = metrics.get(rule['metric'], None)
        return v is not None and (lo <= v <= hi)
    else:
        v = metrics.get(rule['metric'], None)
        if v is None: return False
        op = rule.get('op','>=')
        thresh = rule['value']
        if op == '>=': return v >= thresh
        if op == '<=': return v <= thresh
        if op == '>': return v > thresh
        if op == '<': return v < thresh
        if op == '==': return abs(v - thresh) < 1e-6
        return False

def evaluate_rules(rules:List[Dict], metrics:Dict[str,float], context: Optional[Dict] = None) -> Dict[str,bool]:
    return {r['id']: eval_rule(r, metrics, context=context) for r in rules}
