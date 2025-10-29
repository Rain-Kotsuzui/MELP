import warp as wp

@wp.func
def do(a:wp.vec3f, b:wp.vec3f):
    if a.x < 0:
        return wp.vec3f(a.x + b.x, a.y + b.y, a.z + b.z)
    else:
        return wp.vec3f(a.x - b.x, a.y - b.y, a.z - b.z)
@wp.kernel
def dot(a:wp.array(dtype=wp.vec3f), b:wp.vec3f): #type: ignore
    a[0]=do(a[0], b)
    
a = wp.vec3f(1,2,3)
b= wp.outer(a,a)*a
print(wp.outer(a,a))
print(b)