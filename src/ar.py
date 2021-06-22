import cv2
import numpy as np
import math



def calHomography(marker, dst_pts):
    dst_pts = np.array(dst_pts)
    h, w = marker.shape # (600, 600)
    src_pts = np.array([[0, 0], [h, 0], [h, w], [0, w]])
    H, mask = cv2.findHomography(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    return H

def get_extended_RT(A, H):
    #finds r3 and appends
    # A is the intrinsic mat, and H is the homography estimated
    H = np.float64(H) #for better precision
    A = np.float64(A)
    R_12_T = np.linalg.inv(A).dot(H)

    r1 = np.float64(R_12_T[:, 0]) #col1
    r2 = np.float64(R_12_T[:, 1]) #col2
    T = R_12_T[:, 2] #translation
    
    #ideally |r1| and |r2| should be same
    #since there is always some error we take square_root(|r1||r2|) as the normalization factor
    norm = np.float64(math.sqrt(np.float64(np.linalg.norm(r1)) * np.float64(np.linalg.norm(r2))))
    
    r3 = np.cross(r1,r2)/(norm)
    R_T = np.zeros((3, 4))
    R_T[:, 0] = r1
    R_T[:, 1] = r2 
    R_T[:, 2] = r3 
    R_T[:, 3] = T
    return R_T

def augment(img, obj, projection, template, color=False, scale = 50):
    # takes the captureed image, object to augment, and transformation matrix  
    #adjust scale to make the object smaller or bigger, 4 works for the fox

    h, w = template.shape
    vertices = obj.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)

    #blacking out the aruco marker
    # a = np.array([[0,0,0], [w, 0, 0],  [w,h,0],  [0, h, 0]], np.float64 )
    # imgpts = np.int32(cv2.perspectiveTransform(a.reshape(-1, 1, 3), projection))
    # cv2.fillConvexPoly(img, imgpts, (0,0,0))
    dst_array = []
    #projecting the faces to pixel coords and then drawing
    for face in obj.faces:
        #a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices]) #-1 because of the shifted numbering
        points = scale*points
        points = np.array([[p[2] + w/2, p[0] + h/2, p[1]] for p in points]) #shifted to centre 
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)#transforming to pixel coords

        imgpts = np.int32(dst)
        dst_array.append(imgpts)

        if color is False:
            cv2.fillConvexPoly(img, imgpts, (50, 50, 50))
        else:
            cv2.fillConvexPoly(img, imgpts, face[-1])
    dst_array = np.asarray(dst_array)
    print(dst_array.shape)
    np.save('dst_array_full_hoop.npy', dst_array)
            
    return img

def augment_v2(img, obj, dst_array):

    for e, face in enumerate(obj.faces):
        cv2.fillConvexPoly(img, dst_array[e], face[-1])
            
    return img


class ThreeDimObject:
    def __init__(self, filename_obj, filename_texture, color_fixed = False):
        self.texture = cv2.imread(filename_texture)
        self.vertices = []
        self.faces = []
        #each face is a list of [lis_vertices, lis_texcoords, color]
        self.texcoords = []

        for line in open(filename_obj, "r"):
            if line.startswith('#'): 
                #it's a comment, ignore 
                continue

            values = line.split()
            if not values:
                continue
            
            if values[0] == 'v':
                #vertex description (x, y, z)
                v = [float(a) for a in values[1:4] ]
                self.vertices.append(v)

            elif values[0] == 'vt':
                #texture coordinate (u, v)
                self.texcoords.append([float(a) for a in values[1:3] ])

            elif values[0] == 'f':
                #face description 
                face_vertices = []
                face_texcoords = []
                for v in values[1:]:
                    w = v.split('/')
                    face_vertices.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        face_texcoords.append(int(w[1]))
                    else:
                        color_fixed = True
                        face_texcoords.append(0)
                self.faces.append([face_vertices, face_texcoords])


        for f in self.faces:
            if not color_fixed:
                f.append(self.decide_face_color(f[-1], self.texture, self.texcoords))
            else:
                f.append((50, 50, 50)) #default color

        # cv2.imwrite('texture_marked.png', self.texture)

    def decide_face_color(self, hex_color, texture, textures):
        #doesnt use proper texture
        #takes the color at the mean of the texture coords

        h, w, _ = texture.shape
        col = np.zeros(3)
        coord = np.zeros(2)
        all_us = []
        all_vs = []

        for i in hex_color:
            t = textures[i - 1]
            coord = np.array([t[0], t[1]])
            u , v = int(w*(t[0]) - 0.0001), int(h*(1-t[1])- 0.0001)
            all_us.append(u)
            all_vs.append(v)

        u = int(sum(all_us)/len(all_us))
        v = int(sum(all_vs)/len(all_vs))

        # all_us.append(all_us[0])
        # all_vs.append(all_vs[0])
        # for i in range(len(all_us) - 1):
        #     texture = cv2.line(texture, (all_us[i], all_vs[i]), (all_us[i + 1], all_vs[i + 1]), (0,0,255), 2)
        #     pass    

        col = np.uint8(texture[v, u])
        col = [int(a) for a in col]
        col = tuple(col)
        return (col)