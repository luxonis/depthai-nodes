// Numerically conservative kernels for the frozen geometric Runner.
// Perspective distortion follows OpenCV 4.12's documented rational/tilted model:
// https://github.com/opencv/opencv/blob/4.12.0/modules/calib3d/src/distortion_model.hpp
// https://github.com/opencv/opencv/blob/4.12.0/modules/calib3d/src/calibration_base.cpp
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" int voxel_centroids(const float* xyz, int64_t n, float size,
                               double* centroids, int64_t* membership, const int32_t* rows) {
    try {
        if (n == 0) return 0;
        std::vector<int32_t> grid(3*n);
        int64_t lo[3] = {INT32_MAX, INT32_MAX, INT32_MAX};
        int64_t hi[3] = {INT32_MIN, INT32_MIN, INT32_MIN};
        for (int64_t i=0; i<n; ++i) for (int j=0; j<3; ++j) {
            float q=std::floor(xyz[3*(rows?rows[i]:i)+j]/size);
            if (!std::isfinite(q) || double(q) <= INT32_MIN || double(q) >= INT32_MAX) return -1;
            grid[3*i+j]=(int32_t)q;
            lo[j]=std::min(lo[j],(int64_t)q); hi[j]=std::max(hi[j],(int64_t)q);
        }
        int64_t sy=hi[1]-lo[1]+1, sz=hi[2]-lo[2]+1;
        double volume=double(hi[0]-lo[0]+1)*sy*sz;
        if (volume>8000000) return -1;
        std::vector<int32_t> counts((size_t)volume,0), index((size_t)volume,-1);
        for (int64_t i=0; i<n; ++i) {
            int64_t key=((grid[3*i]-lo[0])*sy+(grid[3*i+1]-lo[1]))*sz+(grid[3*i+2]-lo[2]);
            membership[i]=key; ++counts[key];
        }
        int count=0;
        for (size_t i=0; i<counts.size(); ++i) if (counts[i]) index[i]=count++;
        std::fill(centroids,centroids+3*count,0.0);
        for (int64_t i=0; i<n; ++i) {
            int32_t id=index[membership[i]]; membership[i]=id;
            for (int j=0; j<3; ++j) centroids[3*id+j] += xyz[3*(rows?rows[i]:i)+j];
        }
        for (size_t i=0; i<counts.size(); ++i) if (counts[i])
            for (int j=0; j<3; ++j) centroids[3*index[i]+j] /= counts[i];
        return count;
    } catch (...) { return -1; }
}

extern "C" int height_maxima(const double* xy, const double* heights, int64_t n,
                             double cell, double min_height, double* output) {
    try {
        int64_t lo[2]={INT32_MAX,INT32_MAX}, hi[2]={INT32_MIN,INT32_MIN};
        for (int64_t i=0; i<n; ++i) if (heights[i]>=min_height) for (int j=0; j<2; ++j) {
            double q=std::floor(xy[2*i+j]/cell);
            if (!std::isfinite(q) || q<=INT32_MIN || q>=INT32_MAX) return -1;
            lo[j]=std::min(lo[j],(int64_t)q); hi[j]=std::max(hi[j],(int64_t)q);
        }
        if (hi[0]<lo[0]) return 0;
        int64_t sy=hi[1]-lo[1]+1;
        if (double(hi[0]-lo[0]+1)*sy>8000000) return -1;
        std::vector<double> maxima((hi[0]-lo[0]+1)*sy,-INFINITY);
        for (int64_t i=0; i<n; ++i) if (heights[i]>=min_height) {
            int64_t a=(int64_t)std::floor(xy[2*i]/cell)-lo[0];
            int64_t b=(int64_t)std::floor(xy[2*i+1]/cell)-lo[1];
            maxima[a*sy+b]=std::max(maxima[a*sy+b],heights[i]);
        }
        int count=0;
        for (double v:maxima) if (v!=-INFINITY) output[count++]=v;
        return count;
    } catch (...) { return -1; }
}

// Projection is tiled separately from depth-grid scatter so the arithmetic loop
// can vectorize without changing floating-point contraction or fast-math rules.
template<bool Identity, bool Pinhole>
static void project_cells_impl(const float* __restrict xyz, int64_t n,
                              const double* __restrict t, const double* __restrict k,
                              const double* __restrict intr, const double* __restrict tilt,
                              int width, int height, int cell,
                              int32_t* __restrict cells, float* __restrict depths,
                              const uint8_t* active, float* background) {
    int cols=(width+cell-1)/cell;
    constexpr int tile=1024;
    alignas(64) double us[tile],vs[tile],zs[tile];
    for (int64_t start=0;start<n;start+=tile) {
        int length=(int)std::min((int64_t)tile,n-start);
        #if defined(__clang__)
        #pragma clang loop vectorize(enable)
        #endif
        for (int j=0;j<length;++j) {
            int64_t i=start+j;
            double X=xyz[3*i],Y=xyz[3*i+1],Z=xyz[3*i+2];
            double a,b,z;
            if constexpr(Identity) {a=X;b=Y;z=Z;}
            else {
                a=(X*t[0]+Y*t[1])+Z*t[2]+t[3];
                b=(X*t[4]+Y*t[5])+Z*t[6]+t[7];
                z=(X*t[8]+Y*t[9])+Z*t[10]+t[11];
            }
            double invz=1.0/z,x=a*invz,y=b*invz;
            double r2=x*x+y*y,r4=r2*r2,r6=r4*r2;
            double radial=1+k[0]*r2+k[1]*r4+k[4]*r6;
            double denominator=1+k[5]*r2+k[6]*r4+k[7]*r6;
            double tangent_x=k[2]*(2*x*y)+k[3]*(r2+2*x*x)+k[8]*r2+k[9]*r4;
            double tangent_y=k[2]*(r2+2*y*y)+k[3]*(2*x*y)+k[10]*r2+k[11]*r4;
            double xx=x*radial+tangent_x*denominator;
            double yy=y*radial+tangent_y*denominator;
            // Carry distortion in homogeneous coordinates through sensor tilt.
            // This removes one division per point, retaining double precision.
            double tz=tilt[6]*xx+tilt[7]*yy+tilt[8]*denominator;
            double ti=(tz && denominator) ? 1.0/tz : NAN;
            double dx=ti*(tilt[0]*xx+tilt[1]*yy+tilt[2]*denominator);
            double dy=ti*(tilt[3]*xx+tilt[4]*yy+tilt[5]*denominator);
            if constexpr(Pinhole) {
                us[j]=intr[0]*dx+intr[2]; vs[j]=intr[4]*dy+intr[5];
            } else {
                double u=intr[0]*dx+intr[1]*dy+intr[2];
                double v=intr[3]*dx+intr[4]*dy+intr[5];
                double w=intr[6]*dx+intr[7]*dy+intr[8];
                us[j]=u/w; vs[j]=v/w;
            }
            zs[j]=z;
        }
        for (int j=0;j<length;++j) {
            double u=us[j],v=vs[j],z=zs[j];
            int32_t pixel=(z>0 && std::isfinite(z) && u>=0 && u<width && v>=0 && v<height)
                ? (int)(v/cell)*cols+(int)(u/cell) : -1;
            float depth=(float)z;
            if(active && !active[start+j]) {
                if(pixel>=0) background[pixel]=std::min(background[pixel],depth);
                cells[start+j]=-1;
            } else {cells[start+j]=pixel; depths[start+j]=depth;}
        }
    }
}

extern "C" void project_cells(const float* xyz, int64_t n,
                               const double* t, const double* k, const double* intr,
                               const double* tilt, int width, int height, int cell,
                               int32_t* cells, float* depths, const uint8_t* active, float* background) {
    bool identity=t[0]==1 && t[1]==0 && t[2]==0 && t[3]==0 &&
        t[4]==0 && t[5]==1 && t[6]==0 && t[7]==0 && t[8]==0 && t[9]==0 && t[10]==1 && t[11]==0;
    bool pinhole=intr[1]==0 && intr[3]==0 && intr[6]==0 && intr[7]==0 && intr[8]==1;
    if(identity && pinhole) project_cells_impl<true,true>(xyz,n,t,k,intr,tilt,width,height,cell,cells,depths,active,background);
    else if(pinhole) project_cells_impl<false,true>(xyz,n,t,k,intr,tilt,width,height,cell,cells,depths,active,background);
    else if(identity) project_cells_impl<true,false>(xyz,n,t,k,intr,tilt,width,height,cell,cells,depths,active,background);
    else project_cells_impl<false,false>(xyz,n,t,k,intr,tilt,width,height,cell,cells,depths,active,background);
}

extern "C" void accumulate_grids(const int32_t* cells, const float* depths, int64_t n,
                                  const int32_t* labels, int label_count,
                                  const uint8_t* foreground, int64_t area,
                                  float* grids, float* foreground_grids, const int32_t* included, const float* background) {
    std::fill(grids,grids+area*label_count,INFINITY);
    std::fill(foreground_grids,foreground_grids+area*label_count,INFINITY);
    if(background) std::copy(background,background+area,grids);
    for (int64_t q=0;q<n;++q) {
        int64_t i=included?included[q]:q;
        if(cells[i]<0) continue;
        int64_t pos=(int64_t)labels[i]*area+cells[i];
        grids[pos]=std::min(grids[pos],depths[i]);
        if (foreground[i]) foreground_grids[pos]=std::min(foreground_grids[pos],depths[i]);
    }
}

extern "C" int cross_extrema(const double* xy, int64_t n, int axis, double cell,
                             double* lows, double* highs) {
    try {
        int64_t lo=INT32_MAX, hi=INT32_MIN;
        for (int64_t i=0; i<n; ++i) {
            double q=std::floor(xy[2*i+1-axis]/cell);
            if (!std::isfinite(q) || q<=INT32_MIN || q>=INT32_MAX) return -1;
            lo=std::min(lo,(int64_t)q); hi=std::max(hi,(int64_t)q);
        }
        if (hi-lo+1>8000000) return -1;
        std::vector<double> lower(hi-lo+1,INFINITY), upper(hi-lo+1,-INFINITY);
        for (int64_t i=0; i<n; ++i) {
            int64_t j=(int64_t)std::floor(xy[2*i+1-axis]/cell)-lo;
            lower[j]=std::min(lower[j],xy[2*i+axis]); upper[j]=std::max(upper[j],xy[2*i+axis]);
        }
        int count=0;
        for (size_t j=0;j<lower.size();++j) if (lower[j]!=INFINITY) {
            lows[count]=-lower[j]; highs[count]=upper[j]; ++count;
        }
        return count;
    } catch (...) { return -1; }
}

extern "C" int foreground_rows(const float* xyz, int64_t n, const double* normal,
                                double offset, double low, double high, int32_t* rows) {
    int count=0;
    for (int64_t i=0;i<n;++i) {
        double h=(xyz[3*i]*normal[0]+xyz[3*i+1]*normal[1])+xyz[3*i+2]*normal[2]+offset;
        if (h>=low && h<=high) rows[count++]=i;
    }
    return count;
}

extern "C" void group_rows(const int32_t* rows, const int32_t* labels, int64_t n,
                           const int64_t* starts, int count, int32_t* output) {
    std::vector<int64_t> next(starts,starts+count);
    for (int64_t i=0;i<n;++i) if (next[labels[i]]>=0) output[next[labels[i]]++]=rows[i];
}

static double median_small(double* values, int n) {
    // Windows are tiny (five pixels in the frozen candidate).
    for (int i=1;i<n;++i) { double v=values[i]; int j=i; while(j>0 && values[j-1]>v) {values[j]=values[j-1];--j;} values[j]=v; }
    return n%2 ? values[n/2] : (values[n/2-1]+values[n/2])/2;
}
extern "C" void image_boundary(const uint8_t* image, int h, int w, int channels,
                               int64_t row_stride, int64_t col_stride, int64_t channel_stride,
                               const float* grid, int gh, int gw, const double* samples,
                               int64_t n, const double* outward, const int* params,
                               double contrast, uint8_t* result) {
    const int cell=params[0],margin=params[1],inner=params[2],outer=params[3],band=params[4],continuation=params[5];
    const int first=-band-inner, length=2*band+inner+outer, window=inner+outer;
    for (int64_t i=0;i<n;++i) {
        double reference[3]={0,0,0};
        bool on_parcel=false;
        for (int c=0;c<channels;++c) {
            double values[64]; int count=0;
            for (int offset=first;offset<band+outer;++offset) if (offset<=-cell && offset>-cell-inner) {
                int x=(int)std::nearbyint(samples[2*i]+offset*outward[0]);
                int y=(int)std::nearbyint(samples[2*i+1]+offset*outward[1]);
                if(c==0) {
                    int gx=(int)std::floor((double)x/cell), gy=(int)std::floor((double)y/cell);
                    if(gx>=0 && gx<gw && gy>=0 && gy<gh && std::isfinite(grid[gy*gw+gx])) on_parcel=true;
                }
                x=std::clamp(x,0,w-1); y=std::clamp(y,0,h-1);
                values[count++]=image[(int64_t)y*row_stride+(int64_t)x*col_stride+c*channel_stride];
            }
            reference[c]=count?median_small(values,count):NAN;
        }
        result[i]=0;
        if (!on_parcel) continue;
        bool separates=false;
        // Evaluate the same sliding windows, stopping once their existential test is settled.
        for (int start=0;start<=length-window && !separates;++start) {
            bool valid=true;
            for(int j=0;j<window && valid;++j) {
                int offset=first+start+j;
                int x=(int)std::nearbyint(samples[2*i]+offset*outward[0]);
                int y=(int)std::nearbyint(samples[2*i+1]+offset*outward[1]);
                valid=x>=margin && x<w-margin && y>=margin && y<h-margin;
            }
            if(!valid) continue;
            for(int c=0;c<channels && valid;++c) {
                double values[64];
                for(int j=0;j<inner;++j) {
                    int offset=first+start+j;
                    int x=(int)std::nearbyint(samples[2*i]+offset*outward[0]);
                    int y=(int)std::nearbyint(samples[2*i+1]+offset*outward[1]);
                    values[j]=image[(int64_t)y*row_stride+(int64_t)x*col_stride+c*channel_stride];
                }
                if (!(std::abs(median_small(values,inner)-reference[c])<contrast)) valid=false;
            }
            for(int j=inner;j<window && valid;++j) {
                int offset=first+start+j;
                int x=(int)std::nearbyint(samples[2*i]+offset*outward[0]);
                int y=(int)std::nearbyint(samples[2*i+1]+offset*outward[1]);
                double difference=0;
                for(int c=0;c<channels;++c) difference=std::max(difference,std::abs((double)image[(int64_t)y*row_stride+(int64_t)x*col_stride+c*channel_stride]-reference[c]));
                if (!(difference>=contrast)) valid=false;
            }
            separates=valid;
        }
        if (!separates) continue;
        double exit=INFINITY;
        for (int j=0;j<2;++j) if (outward[j]!=0) {
            double frame=outward[j]>0 ? (j==0?w-1:h-1) : 0;
            exit=std::min(exit,(frame-samples[2*i+j])/outward[j]);
        }
        bool continues=false;
        if (exit>=inner && exit<=continuation) {
            double difference=0;
            for(int c=0;c<channels;++c) {
                double values[64];
                for(int j=0;j<inner;++j) {
                    double offset=std::floor(exit)-j;
                    int x=std::clamp((int)std::nearbyint(samples[2*i]+offset*outward[0]),0,w-1);
                    int y=std::clamp((int)std::nearbyint(samples[2*i+1]+offset*outward[1]),0,h-1);
                    values[j]=image[(int64_t)y*row_stride+(int64_t)x*col_stride+c*channel_stride];
                }
                difference=std::max(difference,std::abs(median_small(values,inner)-reference[c]));
            }
            continues=difference<contrast;
        }
        result[i]=!continues;
    }
}

extern "C" void height_mask(const float* xyz, int64_t n, const double* normal,
                             double offset, double low, uint8_t* mask) {
    for(int64_t i=0;i<n;++i) {
        double h=(xyz[3*i]*normal[0]+xyz[3*i+1]*normal[1])+xyz[3*i+2]*normal[2]+offset;
        mask[i]=h>=low;
    }
}

struct XY { float x,y; };
static double cross(XY a, XY b, XY p) {return (double(b.x)-a.x)*(double(p.y)-a.y)-(double(b.y)-a.y)*(double(p.x)-a.x);}
extern "C" int hull_candidates(const float* xy, int64_t n, float* output) {
    if(n<16) return -1;
    XY extrema[8];
    double best[8]; std::fill(best,best+8,-INFINITY);
    for(int64_t i=0;i<n;++i) {
        XY p={xy[2*i],xy[2*i+1]};
        double v[8]={p.x,p.y,-double(p.x),-double(p.y),double(p.x)+p.y,double(p.x)-p.y,-double(p.x)+p.y,-double(p.x)-p.y};
        for(int j=0;j<8;++j) if(v[j]>best[j]) {best[j]=v[j];extrema[j]=p;}
    }
    std::sort(extrema,extrema+8,[](XY a,XY b){return a.x<b.x || (a.x==b.x && a.y<b.y);});
    XY hull[16]; int k=0;
    for(int i=0;i<8;++i) {while(k>=2 && cross(hull[k-2],hull[k-1],extrema[i])<=0) --k;hull[k++]=extrema[i];}
    int lower=k;
    for(int i=6;i>=0;--i) {while(k>lower && cross(hull[k-2],hull[k-1],extrema[i])<=0) --k;hull[k++]=extrema[i];}
    --k;
    if(k<3) return -1;
    int count=0;
    for(int64_t i=0;i<n;++i) {
        XY p={xy[2*i],xy[2*i+1]}; bool inside=true;
        for(int j=0;j<k && inside;++j) {
            XY a=hull[j],b=hull[(j+1)%k];
            // A conservative positive margin retains near-boundary and collinear points.
            double margin=1e-3*(std::abs(double(b.x)-a.x)+std::abs(double(b.y)-a.y));
            if(cross(a,b,p)<=margin) inside=false;
        }
        if(!inside) {output[2*count]=p.x;output[2*count+1]=p.y;++count;}
    }
    return count;
}

extern "C" int supported_footprint(const float* xyz, int64_t n, const int64_t* membership,
                                   const uint8_t* keep, const double* axes,
                                   const double* normal, double offset,
                                   double* xy, double* heights, double* origin, int fma_mode) {
    int count=0;
    double sx=0,sy=0;
    for(int64_t i=0;i<n;++i) if (keep[membership[i]]) {
        double x=xyz[3*i],y=xyz[3*i+1],z=xyz[3*i+2];
        if(fma_mode) {
            xy[2*count]=std::fma(z,axes[2],std::fma(y,axes[1],x*axes[0]));
            xy[2*count+1]=std::fma(z,axes[5],std::fma(y,axes[4],x*axes[3]));
            heights[count]=(x*normal[0]+y*normal[1])+z*normal[2]+offset;
        } else {
            xy[2*count]=(x*axes[0]+y*axes[1])+z*axes[2];
            xy[2*count+1]=(x*axes[3]+y*axes[4])+z*axes[5];
            heights[count]=(x*normal[0]+y*normal[1])+z*normal[2]+offset;
        }
        sx+=xy[2*count]; sy+=xy[2*count+1];
        ++count;
    }
    origin[0]=sx/count; origin[1]=sy/count;
    return count;
}

extern "C" int centered_hull_candidates(const double* xy, const double* origin, int64_t n, float* output) {
    std::vector<float> centered(2*n);
    for(int64_t i=0;i<n;++i) {centered[2*i]=(float)(xy[2*i]-origin[0]); centered[2*i+1]=(float)(xy[2*i+1]-origin[1]);}
    int count=hull_candidates(centered.data(),n,output);
    if (count<0) {std::copy(centered.begin(),centered.end(),output);return (int)n;}
    return count;
}

extern "C" int rectangle_extrema(const double* xy, const double* origin, const double* axes,
                                 int64_t n, double cell, double* lo0, double* hi0,
                                 double* lo1, double* hi1, double* bounds, int* counts) {
    try {
        std::vector<double> aligned(2*n);
        bounds[0]=bounds[1]=INFINITY; bounds[2]=bounds[3]=-INFINITY;
        for(int64_t i=0;i<n;++i) {
            double x=xy[2*i]-origin[0],y=xy[2*i+1]-origin[1];
            double u=std::fma(y,axes[1],x*axes[0]),v=std::fma(y,axes[3],x*axes[2]);
            aligned[2*i]=u;aligned[2*i+1]=v;
            bounds[0]=std::min(bounds[0],u);bounds[1]=std::min(bounds[1],v);
            bounds[2]=std::max(bounds[2],u);bounds[3]=std::max(bounds[3],v);
        }
        int64_t low_cell[2],high_cell[2];
        for(int j=0;j<2;++j) {
            double lo=std::floor(bounds[j]/cell),hi=std::floor(bounds[j+2]/cell);
            if(!std::isfinite(lo) || !std::isfinite(hi) || lo<=INT32_MIN || hi>=INT32_MAX || hi-lo>8000000) return -1;
            low_cell[j]=(int64_t)lo;high_cell[j]=(int64_t)hi;
        }
        std::vector<double> lower0(high_cell[1]-low_cell[1]+1,INFINITY),upper0(lower0.size(),-INFINITY);
        std::vector<double> lower1(high_cell[0]-low_cell[0]+1,INFINITY),upper1(lower1.size(),-INFINITY);
        for(int64_t i=0;i<n;++i) {
            double u=aligned[2*i],v=aligned[2*i+1];
            int64_t a=(int64_t)std::floor(v/cell)-low_cell[1],b=(int64_t)std::floor(u/cell)-low_cell[0];
            lower0[a]=std::min(lower0[a],u);upper0[a]=std::max(upper0[a],u);
            lower1[b]=std::min(lower1[b],v);upper1[b]=std::max(upper1[b],v);
        }
        counts[0]=counts[1]=0;
        for(size_t j=0;j<lower0.size();++j) if(lower0[j]!=INFINITY) {int q=counts[0]++;lo0[q]=-lower0[j];hi0[q]=upper0[j];}
        for(size_t j=0;j<lower1.size();++j) if(lower1[j]!=INFINITY) {int q=counts[1]++;lo1[q]=-lower1[j];hi1[q]=upper1[j];}
        return 0;
    } catch (...) {return -1;}
}

extern "C" int detect_voxels(const float* xyz, int64_t n, const double* normal, double offset,
                              double low, double high, float size, double* centroids,
                              int64_t* membership, int32_t* rows, int64_t* point_count) {
    try {
        std::vector<int32_t> grid; grid.reserve(3*n);
        int64_t lo[3]={INT32_MAX,INT32_MAX,INT32_MAX},hi[3]={INT32_MIN,INT32_MIN,INT32_MIN};
        int64_t count=0;
        for(int64_t i=0;i<n;++i) {
            double h=(xyz[3*i]*normal[0]+xyz[3*i+1]*normal[1])+xyz[3*i+2]*normal[2]+offset;
            if(h>=low && h<=high) {
                rows[count++]=i;
                for(int j=0;j<3;++j) {
                    float q=std::floor(xyz[3*i+j]/size);
                    if(!std::isfinite(q) || double(q)<=INT32_MIN || double(q)>=INT32_MAX) return -1;
                    grid.push_back((int32_t)q);lo[j]=std::min(lo[j],(int64_t)q);hi[j]=std::max(hi[j],(int64_t)q);
                }
            }
        }
        *point_count=count;
        if(!count) return 0;
        int64_t sy=hi[1]-lo[1]+1,sz=hi[2]-lo[2]+1;
        double volume=double(hi[0]-lo[0]+1)*sy*sz;
        if(volume>8000000) return -1;
        std::vector<int32_t> counts((size_t)volume,0),index((size_t)volume,-1);
        for(int64_t i=0;i<count;++i) {
            int64_t key=((grid[3*i]-lo[0])*sy+(grid[3*i+1]-lo[1]))*sz+(grid[3*i+2]-lo[2]);
            membership[i]=key;++counts[key];
        }
        int voxels=0;
        for(size_t i=0;i<counts.size();++i) if(counts[i]) index[i]=voxels++;
        std::fill(centroids,centroids+3*voxels,0.0);
        for(int64_t i=0;i<count;++i) {
            int32_t id=index[membership[i]];membership[i]=id;
            for(int j=0;j<3;++j) centroids[3*id+j]+=xyz[3*(int64_t)rows[i]+j];
        }
        for(size_t i=0;i<counts.size();++i) if(counts[i])
            for(int j=0;j<3;++j) centroids[3*index[i]+j]/=counts[i];
        return voxels;
    } catch (...) {return -1;}
}
