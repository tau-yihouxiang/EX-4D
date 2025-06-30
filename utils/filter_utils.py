def get_adjacent_faces(faces, face_normals, H, W, threshold=0.8):
    semi_faces_num = faces.shape[0] // 2
    semi_idx = torch.arange(0, semi_faces_num, dtype=torch.int32).to(face_normals.device)
    face_normal1, face_normal2 = face_normals[semi_idx], face_normals[semi_idx + semi_faces_num]
    dot = torch.sum(face_normal1 * face_normal2, dim=-1)
    keep_mask = dot > threshold
    keep_mask = torch.cat([keep_mask, keep_mask], 0)
    return keep_mask
    # # filter first-semi
    # semi_idx = torch.arange(0, semi_faces_num, dtype=torch.int32).to(face_normals.device)
    # adj1 = semi_idx + semi_faces_num
    # adj2 = semi_idx + semi_faces_num - 1
    # adj2[torch.logical_or(adj2 < 0, adj1 % W == 0)] = 0
    # adj3 = semi_idx + semi_faces_num - W
    # adj3[adj3 < 0] = 0
    # semi_normal = face_normals[semi_idx]
    # normal1, normal2, normal3 = face_normals[adj1], face_normals[adj2], face_normals[adj3]
    # dot1, dot2, dot3 = torch.sum(semi_normal * normal1, dim=-1), torch.sum(semi_normal * normal2, dim=-1), torch.sum(semi_normal * normal3, dim=-1)
    # min_dot = torch.minimum(torch.minimum(dot1, dot2), dot3)
    # keep_mask = min_dot > threshold
    # # filter left-semi
    # semi_idx = torch.arange(0, semi_faces_num, dtype=torch.int32).to(face_normals.device) + semi_faces_num
    # adj1 = semi_idx - semi_faces_num
    # adj2 = semi_idx - semi_faces_num + 1
    # adj2[torch.logical_or(adj2 >=semi_faces_num, adj1 % W == 0)] = 0
    # adj3 = semi_idx - semi_faces_num + W
    # adj3[adj3 >= semi_faces_num] = 0
    # semi_normal = face_normals[semi_idx]
    # normal1, normal2, normal3 = face_normals[adj1], face_normals[adj2], face_normals[adj3]
    # dot1, dot2, dot3 = torch.sum(semi_normal * normal1, dim=-1), torch.sum(semi_normal * normal2, dim=-1), torch.sum(semi_normal * normal3, dim=-1)
    # min_dot = torch.minimum(torch.minimum(dot1, dot2), dot3)
    # keep_mask2 = min_dot > threshold
    # keep_mask = torch.cat([keep_mask, keep_mask2], 0)
    return keep_mask