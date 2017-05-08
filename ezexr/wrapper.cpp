#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include <ImathBox.h>
#include <ImfIO.h>
#include <Iex.h>
#include <ImfArray.h>
#include <ImfAttribute.h>
#include <ImfBoxAttribute.h>
#include <ImfChannelList.h>
#include <ImfStandardAttributes.h>
#include <ImfChannelListAttribute.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfCompressionAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfEnvmapAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <ImfIntAttribute.h>
#include <ImfKeyCodeAttribute.h>
#include <ImfLineOrderAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOutputFile.h>
#include <ImfPreviewImageAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfTileDescriptionAttribute.h>
#include <ImfTiledOutputFile.h>
#include <ImfTimeCodeAttribute.h>
#include <ImfVecAttribute.h>
#include <ImfVersion.h>

using namespace std;
using namespace Imf;
using namespace Imath_2_2;


union poutine
{
    Array2D<uint32_t> *u;
    Array2D<half> *h;
    Array2D<float> *f;
};


extern "C" {

    float* readEXRfloat(const char filename[], char **channel_names[], int *width, int *height, int *nb_channels)
    {
        InputFile file(filename);
        Box2i dw = file.header().dataWindow();

        *width = dw.max.x - dw.min.x + 1;
        *height = dw.max.y - dw.min.y + 1;

        FrameBuffer frameBuffer;

        *nb_channels = 0;

        const ChannelList &channels = file.header().channels();

        for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i) {
            (*nb_channels)++;
        }
        *channel_names = new char*[*nb_channels];
        poutine *arrays = new poutine[*nb_channels];

        *nb_channels = 0;
        for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i) {
            const Channel &channel = i.channel();

            (*channel_names)[*nb_channels] = new char[strlen(i.name())];
            strcpy((*channel_names)[*nb_channels], i.name());

            switch (channel.type) {
                case UINT:
                    arrays[*nb_channels].u = new Array2D<uint32_t>;
                    arrays[*nb_channels].u->resizeErase(*height, *width);
                    frameBuffer.insert (i.name(), // name
                                Slice (channel.type, // type
                                    (char *) (arrays[*nb_channels].u[0][0] - // base
                                              dw.min.x - dw.min.y * (*width)),
                                    sizeof (*arrays[*nb_channels].u[0][0]) * 1, // xStride
                                    sizeof (*arrays[*nb_channels].u[0][0]) * (*width), // yStride
                                    1, 1, // x/y sampling
                                    0.0)); // fillValue
                    break;
                case HALF:
                    arrays[*nb_channels].h = new Array2D<half>;
                    arrays[*nb_channels].h->resizeErase(*height, *width);
                    frameBuffer.insert (i.name(), // name
                                Slice (channel.type, // type
                                    (char *) (arrays[*nb_channels].h[0][0] - // base
                                              dw.min.x - dw.min.y * (*width)),
                                    sizeof (*arrays[*nb_channels].h[0][0]) * 1, // xStride
                                    sizeof (*arrays[*nb_channels].h[0][0]) * (*width), // yStride
                                    1, 1, // x/y sampling
                                    0.0)); // fillValue
                    break;
                case FLOAT:
                    arrays[*nb_channels].f = new Array2D<float>;
                    arrays[*nb_channels].f->resizeErase(*height, *width);
                    frameBuffer.insert (i.name(), // name
                                Slice (channel.type, // type
                                    (char *) (arrays[*nb_channels].f[0][0] - // base
                                              dw.min.x - dw.min.y * (*width)),
                                    sizeof (*arrays[*nb_channels].f[0][0]) * 1, // xStride
                                    sizeof (*arrays[*nb_channels].f[0][0]) * (*width), // yStride
                                    1, 1, // x/y sampling
                                    0.0)); // fillValue
                    break;
                default:
                    throw 1;
                    break;
                }

            (*nb_channels)++;
            if (*nb_channels >= 64) { throw 1; }
        }

        file.setFrameBuffer(frameBuffer);
        file.readPixels(dw.min.y, dw.max.y);

        float *retval = (float*)malloc((*nb_channels)*(*width)*(*height)*sizeof(float));

        for (unsigned int k = 0; k < *height; ++k) {
            for (unsigned int j = 0; j < *width; ++j) {
                unsigned int it = 0;
                for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i) {
                    PixelType a = i.channel().type;
                    
                    switch (a) {
                        case UINT:
                            retval[(k*(*width)+j)*(*nb_channels) + it] = (float)(*arrays[it].u)[k][j];
                            break;
                        case HALF:
                            retval[(k*(*width)+j)*(*nb_channels) + it] = (float)(*arrays[it].h)[k][j];
                            break;
                        case FLOAT:
                            retval[(k*(*width)+j)*(*nb_channels) + it] = (*arrays[it].f)[k][j];
                            break;
                        default:
                            throw 1;
                            break;
                    }
                    it++;
                }
            }
        }

        unsigned int it = 0;
        for (ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i) {
            PixelType a = i.channel().type;
            switch (a) {
                case UINT:
                    delete arrays[it].u;
                    break;
                case HALF:
                    delete arrays[it].h;
                    break;
                case FLOAT:
                    delete arrays[it].f;
                    break;
                default:
                    throw 1;
                    break;
            }
            it++;
        }
        return retval;
    }

    void writeEXRfloat(const char filename[], const char *channel_names[], const float *data, int width, int height, int nb_channels)
    {
        Header header (width, height);
        FrameBuffer frameBuffer;

        for (unsigned int i = 0; i < nb_channels; ++i) {
            header.channels().insert (channel_names[i], Channel (FLOAT));
        }

        OutputFile file (filename, header);

        for (unsigned int i = 0; i < nb_channels; ++i) {
            frameBuffer.insert (channel_names[i], // name
                        Slice (FLOAT, // type
                            ((char *) data) + i*height*width*sizeof(*data), // base
                            sizeof (*data) * 1, // xStride
                            sizeof (*data) * (width))); // yStride
        }

        file.setFrameBuffer(frameBuffer);
        file.writePixels(height);

        return;
    }
}
