"""Trace-derived document-retrieval planning diagnostics; no provider logic."""
def document_plan_metrics(trace):
    calls=[e for e in trace if e.get('event')=='tool_requested' and e.get('name')=='search_financial_documents']
    signatures=[(e.get('arguments',{}).get('query'),tuple(sorted((k,v) for k,v in e.get('arguments',{}).items() if k!='query'))) for e in calls]
    return {'document_call_count':len(calls),'one_shot':len(calls)==1,'refinement_attempted':len(calls)>1,
            'identical_repeated_retrieval':len(signatures)!=len(set(signatures)),'exceeded_document_budget':len(calls)>2,
            'materially_changed_query':len(calls)>1 and signatures[0][0]!=signatures[1][0],
            'materially_changed_filters':len(calls)>1 and signatures[0][1]!=signatures[1][1]}
