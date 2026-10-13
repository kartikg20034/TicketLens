from __future__ import annotations
from collections import Counter, defaultdict
from . import vector_store
from .triage import route_ticket

def evaluate(tickets):
    reference=[t for t in tickets if t.get("dataset_split")=="reference"]
    holdout=[t for t in tickets if t.get("dataset_split")=="eval"]
    # Retrieval is always restricted to the reference set by vector_store indexing.
    results=[]; top1=0; top3=0; retrieval_hit=0; confusion=defaultdict(Counter)
    for ticket in holdout:
        similar=vector_store.search(ticket,5)
        prediction=route_ticket(ticket,similar)
        expected=ticket.get("ground_truth_team")
        predicted=prediction.get("team")
        top1 += predicted == expected
        retrieved_teams=[x.get("team") for x in similar]
        top3 += expected in retrieved_teams[:3]
        retrieval_hit += expected in retrieved_teams
        confusion[expected][predicted]+=1
        results.append({"id":ticket["id"],"expected_team":expected,"predicted_team":predicted,"top1":predicted==expected,"retrieved_teams":retrieved_teams,"similar_tickets":similar,"rationale":prediction.get("rationale")})
    n=len(holdout)
    return {"reference_count":len(reference),"holdout_count":n,"top1_accuracy":round(top1/n,4) if n else 0,"retrieval_top3_recall":round(top3/n,4) if n else 0,"retrieval_top5_recall":round(retrieval_hit/n,4) if n else 0,"confusion_matrix":{k:dict(v) for k,v in confusion.items()},"results":results}
