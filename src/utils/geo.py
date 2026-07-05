# src/utils/geo.py
import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from typing import List, Dict, Any

def cluster_by_location(attractions: List[Dict], num_days: int) -> List[List[Dict]]:
    """
    使用K-Means将景点按地理位置分组
    """
    if not attractions:
        return [[] for _ in range(num_days)]
    
    valid = [a for a in attractions if a.get("lat") and a.get("lng")]
    if len(valid) <= num_days:
        groups = [[] for _ in range(num_days)]
        for i, a in enumerate(valid):
            groups[i % num_days].append(a)
        return groups
    
    coords = np.array([[a["lat"], a["lng"]] for a in valid])
    kmeans = KMeans(n_clusters=num_days, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    groups = [[] for _ in range(num_days)]
    for label, a in zip(labels, valid):
        groups[label].append(a)
    
    # 确保没有空组
    for i in range(num_days):
        if not groups[i]:
            max_idx = max(range(num_days), key=lambda j: len(groups[j]))
            if groups[max_idx]:
                groups[i].append(groups[max_idx].pop())
    
    return groups

def optimize_route(attractions: List[Dict]) -> List[Dict]:
    """最近邻算法优化路线顺序"""
    if len(attractions) <= 1:
        return attractions
    
    unvisited = attractions[1:]
    route = [attractions[0]]
    current = attractions[0]
    
    while unvisited:
        idx = min(
            range(len(unvisited)),
            key=lambda i: geodesic(
                (current["lat"], current["lng"]),
                (unvisited[i]["lat"], unvisited[i]["lng"])
            ).km
        )
        route.append(unvisited.pop(idx))
        current = route[-1]
    
    return route
