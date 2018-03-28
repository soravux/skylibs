/* References:
 * https://arxiv.org/abs/1202.6522
*/

#include <math.h>
//#include <stdio.h>


#define M_PI 3.14159265358979323846


inline float amm(float m)
{
    unsigned int k;
    float retval = 1.F;

    for (k=1; k <= m; ++k) {
        retval *= (2.F*k + 1.F) / (2.F*k);
    }
    return sqrt(1.F / (4.F * M_PI) * retval);
}

inline float amn(float m, float n)
{
    return sqrt((4.F*n*n - 1.F) / (n*n - m*m));
}

inline float bmn(float m, float n)
{
    return -sqrt( ((2.F*n + 1.F) / (2.F*n - 3.F)) * (((n - 1.F)*(n - 1.F) - m*m) / (n*n - m*m)) );
}


inline float Pmm(float m, float x)
{
    return amm(m)*pow(1 - x*x, m/2.F);
}

float Pmn(float m, float n, float x)
{
    float res;
    switch ( (int)(n - m) ) {
        case 0:
            //printf("case 0: %f, %f (%f) = %f, %f*%f\n", n, m, x, amm(m)*Pmm(m, x), amm(m), Pmm(m, x));
            // printf("case 0: %f, %f = %f\n", n, m, Pmm(m, x));
            return Pmm(m, x);
        case 1:
            // printf("case 1: %f, %f = %f\n", n, m, amn(m, n)*x*Pmm(m, x));
            return amn(m, n)*x*Pmm(m, x);
        default:
            res = amn(m, n)*x*Pmn(m, n - 1.F, x) + bmn(m, n)*Pmn(m, n - 2.F, x);
            // printf("case other: %f, %f = %f\n", n, m, res);
            return res;
    }
}


// From Robin Green's Gritty Details, http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
// float Pmn(float m, float l, float x)
// {
//     // evaluate an Associated Legendre Polynomial P(l,m,x) at x
//     float pmm = 1.0;
//     if(m>0) {
//         float somx2 = sqrt((1.0-x)*(1.0+x)); 
//         float fact = 1.0;
//         for(int i=1; i<=m; i++) {
//             pmm *= (-fact) * somx2;
//             fact += 2.0;
//         }
//     }
//     if(l==m) { return pmm; }
//     float pmmp1 = x * (2.0*m+1.0) * pmm;
//     if(l==m+1) { return pmmp1; }
//     float pll = 0.0;
//     for(int ll=m+2; ll<=l; ++ll) {
//         pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m);
//         pmm = pmmp1;
//         pmmp1 = pll;
//     }
//     return pll;
// }

void generateAssociatedLegendreFactors(const float N, float *data_out, const float *restrict nodes, const unsigned int num_nodes)
{
    unsigned int i, l, m, j = 0;

    for (i = 0; i < num_nodes; ++i) {
        //printf("New Node: %i\n", i);
        for (l = 0; l < N; ++l) {
            for (m = 0; m <= l; ++m) {
                data_out[j] = Pmn(m, l, nodes[i]);
                // printf("%i: data[%i] = %f\n", i, j, data_out[j]);
                ++j;
            }
        }
    }
    printf("j: %i\n", j);
}
