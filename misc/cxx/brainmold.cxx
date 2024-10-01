#include "ConvertAPI.h"
#include <string>
#include <cstdio>
#include <cmath>
#include <queue>
#include <cstring>
#include <iomanip>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include "SparseMatrix.h"
#include "SparseMatrix.txx"
#include <itksys/SystemTools.hxx>

#include <vnl/vnl_cost_function.h>
#include <vnl/algo/vnl_amoeba.h>

#ifndef M_PI
    #define M_PI   3.14159265358979323846   // pi
#endif

#ifndef M_PI_2
    #define M_PI_2 1.57079632679489661923   // pi/2
#endif

using namespace std;
// using namespace gnlp;

// Define common types
typedef ConvertAPI<double, 3> ConvertAPIType;
typedef typename ConvertAPIType::ImageType ImageType;
typedef typename ImageType::Pointer ImagePointer;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;

class Parameters
{
public:
  // Things related to input and output files
  string fn_input_mri, fn_input_seg, fn_input_dots;
  string fn_input_avoidance;
  string fn_output_root;
  string id_string;

  // Side of the hemisphere
  char side = 'N';

  // Parameters for mold creation
  double slit_spacing_mm = 10.0;
  double slit_width_mm = 1.6;
  double mold_resolution_mm = 0.4;
  double mold_wall_thickness_mm = 3;
  double mold_floor_thickness_mm = 3;

  // The amount of dilation and erosion applied to the mask as a
  // preprocessing step
  int preproc_dilation = 5, preproc_erosion = 3;

  // Flags
  bool flag_print_hemisphere_mold = true;
  bool flag_no_cuts = false;

  // Specific slab to analyze
  int selected_slab = -1;

};

// Kinds of files
enum OutputKind {
  HEMI_MOLD_IMAGE,
  HEMI_VOLUME_IMAGE,
  HEMI_C3D_LOG,
  SLAB_MOLD_IMAGE,
  SLAB_VOLUME_IMAGE,
  SLAB_WITH_DOTS_VOLUME_IMAGE,
  SLAB_C3D_LOG,
  SLAB_OPTIMIZATION_LOG,
  SLAB_CUTPLANE_JSON,
  SLAB_CUT_PNG,
  SLAB_CUT_TEXLET,
  GLOBAL_TEXFILE
};


// Filename generator
string get_output_filename(Parameters &param, OutputKind type, int slab = -1)
{
  const char *root = param.fn_output_root.c_str();
  const char *id = param.id_string.c_str();

  char fn_out[4096];
  switch(type)
  {
    case HEMI_MOLD_IMAGE:
      snprintf(fn_out, 4096, "%s/%s_hemi_mold.nii.gz", root, id);
      break;
    case HEMI_VOLUME_IMAGE:
      snprintf(fn_out, 4096, "%s/%s_hemi_volume.nii.gz", root, id);
      break;
    case HEMI_C3D_LOG:
      snprintf(fn_out, 4096, "%s/logs/%s_hemi_c3d_log.txt", root, id);
      break;
    case SLAB_MOLD_IMAGE:
      snprintf(fn_out, 4096, "%s/%s_slab%02d_mold.nii.gz", root, id, slab);
      break;
    case SLAB_VOLUME_IMAGE:
      snprintf(fn_out, 4096, "%s/%s_slab%02d_volume.nii.gz", root, id, slab);
      break;
    case SLAB_C3D_LOG:
      snprintf(fn_out, 4096, "%s/logs/%s_slab%02d_c3d_log.txt", root, id, slab);
      break;
    case SLAB_OPTIMIZATION_LOG:
      snprintf(fn_out, 4096, "%s/logs/%s_slab%02d_opt_log.txt", root, id, slab);
      break;
    case SLAB_CUTPLANE_JSON:
      snprintf(fn_out, 4096, "%s/%s_slab%02d_cutplanes.json", root, id, slab);
      break;
    case SLAB_CUT_PNG:
      snprintf(fn_out, 4096, "%s/tex/%s_slab%02d_cuts.png", root, id, slab);
      break;
    case SLAB_CUT_TEXLET:
      snprintf(fn_out, 4096, "%s/tex/%s_slab%02d_cuts.tex", root, id, slab);
      break;
    case SLAB_WITH_DOTS_VOLUME_IMAGE:
      snprintf(fn_out, 4096, "%s/%s_slab%02d_mask_with_dots.nii.gz", root, id, slab);
      break;
    case GLOBAL_TEXFILE:
      snprintf(fn_out, 4096, "%s/tex/%s_print_template.tex", root, id);
      break;
  }

  // Wrap in string
  string fn_string(fn_out);

  // Create the directory if it does not exist
  string dir = itksys::SystemTools::GetFilenamePath(fn_string);
  itksys::SystemTools::MakeDirectory(dir);

  // Return the file
  return fn_string;
}


struct Slab {
public:
  double y0, y1;
  ImagePointer i_mask;
};



int usage()
{
  printf(
    "brainmold: Generator for 3D printed brain molds \n"
    "usage:\n"
    "  brainmold [options]\n"
    "required options:\n"
    "  -s                       Subject ID, used in naming files\n"
    "  -h <L|R>                 Which hemisphere (left/right)\n"
    "  -i <mri> <mask>          Input MRI and brain mask\n"
    "  -o <dir>                 Output directory\n"
    "optional inputs:\n"
    "  -d <image>               Dots segmentation image\n"
    "  -a <image>               Avoidance segmentation image\n"
    "  -C                       No cuts/molds for the slabs (new way)\n"
    "  --no-hemi-mold           Skip hemisphere mold printing\n"
    "  --slab <index>           Only generate for one slab (negative for none)\n"
    );

  return -1;
}

template <typename TImage>
void save_image(TImage *img, const std::string &fn)
{
  typedef itk::ImageFileWriter<TImage> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(img);
  writer->SetFileName(fn.c_str());
  writer->Update();
}


/**
 * Representation of a cut plane: (theta/R) pair
 */
typedef std::pair<double, double> CutPlane;
typedef std::vector<CutPlane> CutPlaneList;


/** Get a label at a coordinate based on cut planes (first label is 1) */
int get_block_label(double x, double y, const CutPlaneList &cpl)
{
  int label = 1;
  for(unsigned int k=0; k < cpl.size(); k++)
    {
    int label_k = 1 << k;
    double cos_theta = cos(cpl[k].second);
    double sin_theta = sin(cpl[k].second);
    double r = cpl[k].first;

    double d = x * cos_theta + y * sin_theta;
    label += (d < r) ? label_k : 0;
    }

  return label;
}


/*
 1  procedure BFS(G, root) is
 2      let Q be a queue
 3      label root as discovered
 4      Q.enqueue(root)
 5      while Q is not empty do
 6          v := Q.dequeue()
 7          if v is the goal then
 8              return v
 9          for all edges from v to w in G.adjacentEdges(v) do
10              if w is not labeled as discovered then
11                  label w as discovered
12                  Q.enqueue(w)
*/



int count_connected_components_bfs(
    AbstractImmutableSparseArray *graph,
    int *vertex_label, int *vertex_comp)
{
  std::queue<int> Q;
  int nv = (int) graph->GetNumberOfRows();
  for(int i = 0; i < nv; i++)
    vertex_comp[i] = -1;

  // Call DFS for each vertex
  int comp_id = -1;
  for(int i = 0; i < nv; i++)
    {
    if(vertex_comp[i] < 0)
      {
      // We found the next root, associated with a new component
      int comp_size = 1;
      vertex_comp[i] = ++comp_id;
      Q.push(i);
      while(!Q.empty())
        {
        int j = Q.front(); Q.pop();
        for(auto k = graph->GetRowIndex()[j]; k < graph->GetRowIndex()[j+1]; k++)
          {
          auto m = graph->GetColIndex()[k];
          if(vertex_comp[m] < 0 && vertex_label[j] == vertex_label[m])
            {
            vertex_comp[m] = comp_id; comp_size++;
            Q.push(m);
            }
          }
        }
      }
    }

  return (1+comp_id);
}


class VoxelGraph
{
public:
  typedef ImmutableSparseMatrix<double> SparseMatrixType;

  // Set up the graph
  VoxelGraph(ImageType *img)
  {
    // On the first pass, label the image with vertex indices and set coord arrays
    typedef itk::Image<int, 3> LabelImage;
    typename LabelImage::Pointer imgLabel = LabelImage::New();
    imgLabel->CopyInformation(img);
    imgLabel->SetRegions(img->GetBufferedRegion());
    imgLabel->Allocate();
    imgLabel->FillBuffer(-1);

    auto region = img->GetBufferedRegion();
    itk::ImageRegionConstIteratorWithIndex<ImageType> itSrc(img, region);
    itk::ImageRegionIterator<LabelImage> itDest(imgLabel, region);
    unsigned int n = 0;
    for( ; !itSrc.IsAtEnd(); ++itSrc, ++itDest)
      {
      if(itSrc.Get() > 0)
        itDest.Set(n++);
      }

    // Second pass, generate coordinate arrays
    m_X.set_size(n, 3);
    itk::ImageRegionConstIteratorWithIndex<ImageType> itSrc2(img, region);
    int k = 0;
    for(; !itSrc2.IsAtEnd(); ++itSrc2)
      {
      if(itSrc2.Get())
        {
        // Map index to RAS coordinate
        auto idx = itSrc2.GetIndex();
        itk::Point<double, 3> pt;
        img->TransformIndexToPhysicalPoint(idx, pt);
        pt[0] *= -1.0; pt[1] *= -1.0;

        // Set coordinate array
        for(unsigned int d = 0; d < 3; d++)
          m_X(k,d) = pt[d];

        k++;
        }
      }

    // Compute center of mass, extents
    for(unsigned int d = 0; d < 3; d++)
      {
      auto c = m_X.get_column(d);
      m_XMin[d] = c.min_value();
      m_XMax[d] = c.max_value();
      m_XDim[d] = m_XMax[d] - m_XMin[d];
      m_XCtr[d] = 0.5 * (m_XMax[d] + m_XMin[d]);
      }

    // Shrink region by one for next application
    for(unsigned int d = 0; d < 3; d++)
      region.SetSize(d, region.GetSize()[d] - 1);

    vnl_sparse_matrix<double> smat(n, n);
    itk::ImageRegionIteratorWithIndex<LabelImage> it(imgLabel, region);
    for( ; !it.IsAtEnd(); ++it)
      {
      int l = it.Get();
      if(l >= 0)
        {
        auto idx = it.GetIndex();
        for(unsigned int d = 0; d < 3; d++)
          {
          idx[d]++;
          int l_nbr = imgLabel->GetPixel(idx);
          if(l_nbr >= 0)
            {
            smat(l, l_nbr) = 1.0;
            smat(l_nbr, l) = 1.0;
            }
          idx[d]--;
          }
        }
      }

    // Set the sparse matrix
    m_S.SetFromVNL(smat);

    // Allocate the label array
    m_NodeLabel.set_size(n);
    m_NodeLabel.fill(0);
    m_CompIndex.set_size(n);

    // Compute number of connected components
    m_WholeComp = count_connected_components_bfs(&m_S, m_NodeLabel.data_block(), m_CompIndex.data_block());
  }

  // Apply a set of cut planes to the image
  void ApplyCutPlanes(CutPlaneList &cpl)
  {
    // Reset the distance
    m_MinDistanceToCut = 1e100;

    // Apply each plane to every vertex
    m_NodeLabel.fill(0);
    for(unsigned int k = 0; k < cpl.size(); k++)
      {
      int label_k = 1 << k;
      double cos_theta = cos(cpl[k].second);
      double sin_theta = sin(cpl[k].second);
      double r = cpl[k].first;

      for(unsigned int i = 0; i < m_X.rows(); i++)
        {
        double d = m_X[i][0] * cos_theta + m_X[i][2] * sin_theta;
        m_NodeLabel[i] += (d < r) ? label_k : 0;

        double dd = fabs(d - r);
        if(m_MinDistanceToCut > dd)
          m_MinDistanceToCut = dd;
        }
      }
  }

  // Compute block extents and sizes with respect to given orientation
  void ComputeBlockExtentsAndSizes(int n_planes, double phi)
  {
    // Compute extents and sizes
    unsigned int nr = 1 << n_planes;
    double cos_phi = cos(phi), sin_phi = sin(phi);
    m_BlockExtents.set_size(nr, 4); m_BlockExtents.fill(0.0);
    m_BlockSizes.set_size(nr); m_BlockSizes.fill(0.0);
    for(unsigned int i = 0; i < m_X.rows(); i++)
      {
      int l = m_NodeLabel[i];
      double u = m_X[i][0] * cos_phi + m_X[i][2] * sin_phi;
      double v = -m_X[i][0] * sin_phi + m_X[i][2] * cos_phi;
      if(m_BlockSizes[l] == 0 || u < m_BlockExtents(l, 0)) m_BlockExtents(l, 0) = u;
      if(m_BlockSizes[l] == 0 || u > m_BlockExtents(l, 1)) m_BlockExtents(l, 1) = u;
      if(m_BlockSizes[l] == 0 || v < m_BlockExtents(l, 2)) m_BlockExtents(l, 2) = v;
      if(m_BlockSizes[l] == 0 || v > m_BlockExtents(l, 3)) m_BlockExtents(l, 3) = v;
      m_BlockSizes[l]++;
      }

    // Normalize the volumes
    m_BlockRelativeSizes = m_BlockSizes * (1.0 / m_BlockSizes.sum());
  }

  // Get the number of components
  int CountComponents()
  {
    return count_connected_components_bfs(&m_S, m_NodeLabel.data_block(), m_CompIndex.data_block());
  }


  // Sparse matrix representation of non-zero voxels
  SparseMatrixType m_S;

  // Matrix of xyz coordinates of the foreground voxels in the mask
  vnl_matrix<double> m_X, m_DotsX, m_BlockExtents;
  vnl_vector<double> m_BlockSizes;
  vnl_vector<double> m_BlockRelativeSizes;

  // Center of mass
  vnl_vector_fixed<double, 3> m_XMin, m_XMax, m_XDim, m_XCtr;

  // Node labels and connected component labels
  vnl_vector<int> m_NodeLabel, m_CompIndex;

  double m_MinDistanceToCut;

  // Number of components
  int m_WholeComp;
};


class SlabCutPlaneBruteOptimization
{
public:
  typedef ImmutableSparseMatrix<double> SparseMatrixType;

  /** Initialize graph data structures for an input slab */
  SlabCutPlaneBruteOptimization(ImageType *slab_full, ImageType *dots, ImageType *avoid)
    : m_VGFull(slab_full), m_VGDots(dots), m_VGAvoid(avoid)
  {
    // Output goes to std::out
    m_SOut = &std::cout;
  }

  /** Set up optimization for specific number of cut planes */
  void SetupOptimization(int extra_cuts_w, int extra_cuts_h)
  {
    // Figure out the minimum number of cuts (based on slide dimensions)
    m_NumCutsW = (int) (m_VGFull.m_XDim[0] / 70);
    m_NumCutsH = (int) (m_VGFull.m_XDim[2] / 45);

    // Number of cuts
    m_NumCutsW += extra_cuts_w;
    m_NumCutsH += extra_cuts_h;

    // The unknown parameters are the heights/widths of the cuts and orientation
    m_NumUnknowns = 1 + m_NumCutsW + m_NumCutsH;

    // Reset iteration
    m_Iter = 0;
  }

  /** Parse a flat array of parameters into a data structure */
  CutPlaneList GetCutPlanes(const double *x)
  {
    // The parameters are the orientation of the cut planes (theta) and
    // the signed distance of each cut plane from the center of the slide,
    // which is also the center of rotation. From these data, we put build
    // a polar representation of each plane
    CutPlaneList cpl;

    double theta = x[0];

    // Horizontal cuts have same orientation as theta
    for(unsigned int i = 0; i < m_NumCutsW; i++)
      {
      // Adjusting the R for the origin
      double r_i = x[i+1] + cos(theta) * m_VGFull.m_XCtr[0] + cos(theta) * m_VGFull.m_XCtr[2];
      double theta_i = theta;
      cpl.push_back(std::make_pair(r_i, theta_i));
      }

    // Vertical cuts have opposite orientation as theta
    for(unsigned int i = m_NumCutsW; i < m_NumCutsH+m_NumCutsW; i++)
      {
      // Adjusting the R for the origin
      double r_i = x[i+1] - sin(theta) * m_VGFull.m_XCtr[0] + cos(theta) * m_VGFull.m_XCtr[2];
      double theta_i = M_PI_2 + theta;
      cpl.push_back(std::make_pair(r_i, theta_i));
      }

    return cpl;
  }

  /** Compute the objective function */
  double ComputeObjective(const double *x)
  {
    // Get the cut planes
    CutPlaneList cpl = GetCutPlanes(x);

    // Get the global orientation
    double phi = x[0];

    *m_SOut << "==== ITERATION " << setw(3) << ++m_Iter << " ====" << endl;
    *m_SOut << "  Parameters: " << vnl_vector<double>(x, m_NumUnknowns) << endl;

    // Apply cuts to the two graphs
    m_VGFull.ApplyCutPlanes(cpl);
    m_VGDots.ApplyCutPlanes(cpl);
    m_VGAvoid.ApplyCutPlanes(cpl);

    // Now compute statistics for each cut (extents, volume, etc);
    m_VGFull.ComputeBlockExtentsAndSizes(cpl.size(), phi);

    // Compute penalty terms
    double penalty_extent = 0;
    double sum_vol = 0, ssq_vol = 0;
    int n_vol = 0;
    for(unsigned int l = 0; l < m_VGFull.m_BlockExtents.rows(); l++)
      {
      if(m_VGFull.m_BlockRelativeSizes[l] > 0)
        {
        double w = m_VGFull.m_BlockExtents(l,1) - m_VGFull.m_BlockExtents(l,0);
        double h = m_VGFull.m_BlockExtents(l,3) - m_VGFull.m_BlockExtents(l,2);
        double ext_min = min(w, h), ext_max = max(w, h);

        char buffer[256];
        snprintf(buffer, 256, "  Piece %d:  Extent: %4.2f by %4.2f  Rel.Size = %6.4f",
                l, ext_max, ext_min, m_VGFull.m_BlockRelativeSizes[l]);
        *m_SOut << buffer<< endl;

        if(ext_max > 74.)
          penalty_extent += 100000 * (ext_max - 74.);
        if(ext_min > 49.)
          penalty_extent += 100000 * (ext_min - 49.);
        sum_vol += m_VGFull.m_BlockRelativeSizes[l];
        ssq_vol += m_VGFull.m_BlockRelativeSizes[l] * m_VGFull.m_BlockRelativeSizes[l];
        n_vol++;
        }
      }
    *m_SOut << "  Extent violation penalty: " << penalty_extent << endl;

    // Compute number of connected components
    int n_comp_full = m_VGFull.CountComponents();
    double penalty_conn = n_comp_full * 1.0;

    *m_SOut << "  Connected components in full graph: " << n_comp_full << endl;
    *m_SOut << "  Connected components penalty: " << penalty_conn << endl;

    // Compute the minimum distance from dots to cut planes

    // Apply each plane to every vertex
    double min_dots_distance = m_VGDots.m_MinDistanceToCut;
    double min_avoid_distance = m_VGAvoid.m_MinDistanceToCut;

    double dots_dist_penalty = (min_dots_distance < 5) ? 100 * (5 - min_dots_distance) : 0;
    double avoid_dist_penalty = (min_avoid_distance < 5) ? 10 * (5 - min_avoid_distance) : 0;

    *m_SOut << "  Minimum distance from dots to cutplanes: " << min_dots_distance << endl;
    *m_SOut << "  Dots distance penalty: " << dots_dist_penalty << endl;
    *m_SOut << "  Minimum distance from dots to avoidance regions: " << min_avoid_distance << endl;
    *m_SOut << "  Avoidance distance penalty: " << avoid_dist_penalty << endl;

    // Compute the variance in volume
    double var_vol = (ssq_vol - sum_vol / n_vol) / n_vol;
    double std_vol = sqrt(var_vol);
    *m_SOut << "  Volume variance: " << std_vol << endl;
    *m_SOut << "  Volume variance penalty:" << std_vol << endl;

    // Total objective
    double total_obj = penalty_extent + penalty_conn + dots_dist_penalty + avoid_dist_penalty + var_vol;
    *m_SOut << "  Total objective:" << total_obj << endl;

    return total_obj;
  }

  class CostFunction : public vnl_cost_function
  {
  public:
    CostFunction(SlabCutPlaneBruteOptimization *self) : m_Parent(self) {}
    virtual double f(vnl_vector<double> const& x) { return m_Parent->ComputeObjective(x.data_block()); }
  private:
    SlabCutPlaneBruteOptimization *m_Parent;
  };


  /** Perform optimization */
  vnl_vector<double> Optimize(unsigned int n_iter)
  {
    CostFunction cfun(this);
    vnl_amoeba alg(cfun);
    vnl_vector<double> x = this->GetInitialParameters();
    vnl_vector<double> dx(x.size(), 5.0);
    dx[0] = M_PI / 18;
    alg.set_max_iterations(n_iter);
    alg.minimize(x, dx);

    return x;
  }

  /** Generate initial parameters */
  vnl_vector<double> GetInitialParameters()
  {
    vnl_vector<double> ip(m_NumUnknowns);

    // Initialize theta to zero
    ip[0] = 0;

    // Equally space out in width and height
    for(int i = 0; i < m_NumCutsW; i++)
      {
      ip[i+1] = (i+1) * m_VGFull.m_XDim[0] / (m_NumCutsW+1.) - m_VGFull.m_XDim[0] / 2.0;
      }

    for(int i = 0; i < m_NumCutsH; i++)
      {
      ip[i+m_NumCutsW+1] = (i+1) * m_VGFull.m_XDim[2] / (m_NumCutsH+1.) - m_VGFull.m_XDim[2] / 2.0;
      }

    return ip;
  }

  void RedirectOutput(std::ostream &sout) { m_SOut = &sout; }

  int GetNumberOfComponentsInSlab() const { return m_VGFull.m_WholeComp; }

private:

  // Voxel graphs for the full image and eroded image
  VoxelGraph m_VGFull, m_VGDots, m_VGAvoid;

  // Matrix of xyz coordinates of the foreground voxels in the mask
  vnl_matrix<double> m_DotsX;

  // Number of cuts for current optimization setup
  int m_NumCutsW, m_NumCutsH, m_NumUnknowns, m_Iter;

  // Output stream
  std::ostream *m_SOut;
};

// A nice palette generated by https://mokole.com/palette.html
int palette[] = {
  0x66cdaa, // mediumaquamarine
  0xda70d6, // orchid
  0xffd700, // gold
  0x4169e1, // royalblue
  0x2f4f4f, // darkslategray
  0x556b2f, // darkolivegreen
  0x8b4513, // saddlebrown
  0x191970, // midnightblue
  0x708090, // slategray
  0x8b0000, // darkred
  0x808000, // olive
  0x483d8b, // darkslateblue
  0x5f9ea0, // cadetblue
  0x008000, // green
  0x3cb371, // mediumseagreen
  0xbc8f8f, // rosybrown
  0x663399, // rebeccapurple
  0xb8860b, // darkgoldenrod
  0xbdb76b, // darkkhaki
  0xcd853f, // peru
  0x4682b4, // steelblue
  0x000080, // navy
  0xd2691e, // chocolate
  0x9acd32, // yellowgreen
  0x20b2aa, // lightseagreen
  0xcd5c5c, // indianred
  0x32cd32, // limegreen
  0x7f007f, // purple2
  0x8fbc8f, // darkseagreen
  0xb03060, // maroon3
  0xd2b48c, // tan
  0x9932cc, // darkorchid
  0xff0000, // red
  0xffa500, // orange
  0xffff00, // yellow
  0xc71585, // mediumvioletred
  0x0000cd, // mediumblue
  0x7cfc00, // lawngreen
  0x00ff00, // lime
  0x00fa9a, // mediumspringgreen
  0xe9967a, // darksalmon
  0xdc143c, // crimson
  0x00ffff, // aqua
  0x00bfff, // deepskyblue
  0x9370db, // mediumpurple
  0x0000ff, // blue
  0xa020f0, // purple3
  0xff6347, // tomato
  0xd8bfd8, // thistle
  0xff00ff, // fuchsia
  0x1e90ff, // dodgerblue
  0xdb7093, // palevioletred
  0xf0e68c, // khaki
  0xdda0dd, // plum
  0x87ceeb, // skyblue
  0xff1493, // deeppink
  0xafeeee, // paleturquoise
  0xfaf0e6, // linen
  0x98fb98, // palegreen
  0x7fffd4, // aquamarine
  0xff69b4, // hotpink
  0xfffacd, // lemonchiffon
  0xffb6c1, // lightpink
  0xa9a9a9 // darkgray
};

struct LabelStats {
  int label;
  int count = 0;
  vnl_vector_fixed<double, 3> center_of_mass, extent, x_min, x_max;
};

map<int, LabelStats> get_label_stats(ImageType *img)
{
  map<int, LabelStats> stats;
  itk::ImageRegionIteratorWithIndex<ImageType> iter(img, img->GetBufferedRegion());
  LabelStats *curr_stat = NULL;
  for(; !iter.IsAtEnd(); ++iter)
    {
    // Round the double value to an int
    int label = (int) (iter.Get() + 0.5);
    if(label <= 0)
      continue;

    // Point to the current stat
    if(curr_stat == NULL || curr_stat->label != label)
      curr_stat = &(stats[label]);

    // Get coordinate
    typename ImageType::PointType pt;
    img->TransformIndexToPhysicalPoint(iter.GetIndex(), pt);
    pt[0] *= -1.0; pt[1] *= -1.0;
    vnl_vector_fixed<double, 3> x_pt(pt.GetVnlVector().data_block());

    // Initialize stat
    if(curr_stat->count == 0)
      {
      curr_stat->label = label;
      curr_stat->center_of_mass = x_pt;
      curr_stat->x_min = x_pt;
      curr_stat->x_max = x_pt;
      curr_stat->count = 1;
      }
    else
      {
      curr_stat->center_of_mass += x_pt;
      for(unsigned int d = 0; d < 3; d++)
        {
        curr_stat->x_min[d] = min(curr_stat->x_min[d], x_pt[d]);
        curr_stat->x_max[d] = max(curr_stat->x_max[d], x_pt[d]);
        }
      curr_stat->count++;
      }
    }

  // Normalize
  for(auto &x : stats)
    {
    x.second.center_of_mass *= 1.0 / x.second.count;
    x.second.extent = x.second.x_max - x.second.x_min;
    }

  return stats;
}

/**
 * Simplified version of this code where we just output slabs with dots and generate the
 * PDF in Python
 */
void process_slab_nocuts(
    Parameters &param, int slab_id, Slab &slab,
    ImageType *i_hemi_raw, ImageType *i_hemi_mask, ImageType *i_dots, ImageType *i_avoid)
{
  // C3D output should be piped to a log file
  ofstream c3d_log(get_output_filename(param, SLAB_C3D_LOG, slab_id).c_str());

  // Create our own API
  ConvertAPIType api;
  api.RedirectOutput(c3d_log, c3d_log);

  // Deal with the raw image (for visualization)
  api.AddImage("H", i_hemi_mask);

  // Extract the slab image region
  api.Execute(
        "-verbose -clear -push H -cmp -pick 1 -thresh %f %f 255 0 "
        "-push H -times -thresh 0 127 0 255 -info -trim 0vox -as slab ",
        slab.y0, slab.y1);

  // Check if the image is empty
  auto i_slab = api.GetImage("slab");
  if(i_slab->GetBufferedRegion().GetNumberOfPixels() == 0)
    {
    cout << "  Empty slab encountered" << endl;
    return;
    }

  // Get all the dots and render them in the slab space
  struct DotStat { int count = 0; itk::Point<double, 3> p; };
  std::map<int, DotStat > dot_stats;
  itk::ImageRegionIteratorWithIndex<ImageType> itd(i_dots, i_dots->GetBufferedRegion());
  for(; !itd.IsAtEnd(); ++itd)
    {
    if(itd.Value() != 0)
      {
      itk::Point<double, 3> p;
      i_dots->TransformIndexToPhysicalPoint(itd.GetIndex(), p);
      dot_stats[itd.Value()].count++;
      for(unsigned int j = 0; j < 3; j++)
        dot_stats[itd.Value()].p[j] += p[j];
      }
    }

  // Render each dot into the slab
  for(auto &it : dot_stats)
    {
    // Find the center of the dot
    for(unsigned int j = 0; j < 3; j++)
      it.second.p[j] = it.second.p[j] / it.second.count;

    // Project into the space of the slab image
    itk::Index<3> idx_center;
    i_slab->TransformPhysicalPointToIndex(it.second.p, idx_center);

    // Create a region around the center - a bit larger and inside of the slab
    itk::ImageRegion<3> dot_region(idx_center, {1u,1u,1u});
    dot_region.PadByRadius(1);
    if(!dot_region.Crop(i_slab->GetBufferedRegion()))
      continue;

    // Fill the region with this dot
    if(dot_region.GetNumberOfPixels() > 0)
      {
      printf("Slab %d contains dot %d with physical coordinates (%8.4f, %8.4f, %8.4f)\n",
             slab_id, it.first, it.second.p[0], it.second.p[1], it.second.p[2]);
      }
    for(itk::ImageRegionIterator<ImageType> itdot(i_slab, dot_region);
        !itdot.IsAtEnd(); ++itdot)
      {
      itdot.Set(it.first);
      }
    }

  // Save the image
  api.Execute("-push slab -type uchar -o %s",
              get_output_filename(param, SLAB_WITH_DOTS_VOLUME_IMAGE, slab_id).c_str());

}

void process_slab(Parameters &param, int slab_id, Slab &slab,
                  ImageType *i_hemi_raw, ImageType *i_hemi_mask, ImageType *i_dots, ImageType *i_avoid)
{
  // C3D output should be piped to a log file
  ofstream c3d_log(get_output_filename(param, SLAB_C3D_LOG, slab_id).c_str());
  ofstream o_json(get_output_filename(param, SLAB_CUTPLANE_JSON, slab_id));

  // Create our own API
  ConvertAPIType api;
  api.RedirectOutput(c3d_log, c3d_log);

  // Deal with the raw image (for visualization)
  api.AddImage("raw", i_hemi_raw);
  api.Execute("-verbose -push raw -cmp -pick 1 -thresh %f %f 1 0 -as slab_mask "
              "-push raw -times -trim 10x2x10vox -as slab_raw -o /tmp/slab_raw.nii.gz",
              slab.y0, slab.y1);

  // Load the mask image
  api.AddImage("H", i_hemi_mask);

  // Figure out the size of the mean filter for averaging the slab
  int mf_rad = (int) ceil((slab.y1 - slab.y0) / (2 * param.mold_resolution_mm));

  // The padding applied is related to floor thickness
  int y_pad = (int) ceil(param.mold_floor_thickness_mm / param.mold_resolution_mm);

  // Extract the slab image region
  api.Execute(
        "-clear -push H -cmp -pick 1 -thresh %f %f 1 0 "
        "-push H -times -trim %dvox -as slab ",
        slab.y0, slab.y1, y_pad, mf_rad);

  // For the cut plane computation, apply more some erosion to the slab, so that
  // we don't end up with really tiny tissue bridges
  api.Execute("-dilate 0 11x0x11 -as slab_trim");

  // Extract the slab from the dots image as well
  api.AddImage("dots", i_dots);
  api.Execute(
        "-clear -push dots -cmp -pick 1 -thresh %f %f 1 0 -push dots -times "
        "-insert slab 1 -int 0 -reslice-identity -as dots_slab");

  // Extract the slab from the avoidance image as well
  api.AddImage("avoid", i_avoid);
  api.Execute(
        "-clear -push avoid -cmp -pick 1 -thresh %f %f 1 0 -push avoid -times "
        "-insert slab 1 -int 0 -reslice-identity -as avoid_slab");

  // Cut plane list
  CutPlaneList cpl;

  // Optimization output should be piped to a log file
  string fn_opt_log = get_output_filename(param, SLAB_OPTIMIZATION_LOG, slab_id);
  ofstream opt_log(fn_opt_log.c_str());

  // Set up the optimizer
  SlabCutPlaneBruteOptimization scp_opt(api.GetImage("slab"), api.GetImage("dots_slab"), api.GetImage("avoid_slab"));
  if(scp_opt.GetNumberOfComponentsInSlab() == 0)
    {
    cout << "  No components in slab, skipping mold generation" << endl;
    return;
    }
  else
    {
    cout << "  Slab contains " << scp_opt.GetNumberOfComponentsInSlab() << " components." << endl;
    }
  scp_opt.RedirectOutput(opt_log);
  scp_opt.SetupOptimization(0,0);
  vnl_vector<double> x0 = scp_opt.GetInitialParameters();
  scp_opt.ComputeObjective(x0.data_block());
  vnl_vector<double> x_opt = scp_opt.Optimize(200);
  cpl = scp_opt.GetCutPlanes(x_opt.data_block());

  // If there is no cuts, return
  if(cpl.size() == 0)
    {
    cout << "  No cutting needed for this slab." << endl;
    o_json << "[" << endl << "]" << endl;
    return;
    }

  // Generate a mold for this slab
  cout << "  Generating mold" << endl;

  // Compute coordinate maps and clear the stack
  api.Execute("-push slab -cmp -popas z -popas y -popas x -clear");

  // Apply each cutplane
  for(auto &cp : cpl)
    {
    api.Execute(
          "-push x -scale %f -push z -scale %f -add -shift %f -abs "
          "-stretch 0 %f -4 4 -clip -4 4 ",
          cos(cp.second), sin(cp.second), -cp.first, 2 * param.slit_width_mm);
    }

  // Combine the cut planes into one image
  if(cpl.size() > 1)
    api.Execute("-accum -min -endaccum");

  // Figure out where the base starts: 5vox from the bottom or top.
  api.Execute(" -as cuts -push y -thresh %f inf 4 -4 -max -as cutsbase ", slab.y1);

  // Apply extrusion to the slab itself
  int dilation = (int) ceil(param.mold_wall_thickness_mm / param.mold_resolution_mm);
  api.Execute(
        "-push slab -thresh -inf 0.5 4 -4 -swapdim ARS -extrude-seg -swapdim LPI -as carve1 "
        "-dup -swapdim PRS -extrude-seg -thresh -inf 0 0 1 -dilate 0 0x%dx%d "
        "-stretch 0 1 4 -4 -swapdim LPI -min "
        "-push cutsbase -min -as carved -o %s "
        "-thresh 0.0 inf 1 0 -pad 2x2x2 2x2x2 0 ",
        dilation, dilation,
        get_output_filename(param, SLAB_MOLD_IMAGE, slab_id).c_str());

  cout << "  Generating 3D image of cut regions" << endl;

  // Generate an image of separate chunks
  api.Execute("-clear");
  for(int i = 0; i < cpl.size(); i++)
    {
    auto &cp = cpl[i];
    api.Execute(
          "-push x -scale %f -push z -scale %f -add -shift %f "
          "-thresh -inf 0 %d 0 ",
          cos(cp.second), sin(cp.second), -cp.first, 1 << i);
    }

  if(cpl.size() > 1)
    api.Execute("-accum -add -endaccum");

  api.Execute("-shift 1 -popas part "
              "-push slab_raw -dup -push part -int 0 -reslice-identity -times "
              "-type uchar -as slabvol -o %s",
              get_output_filename(param, SLAB_VOLUME_IMAGE, slab_id).c_str());

  // Generate a JSON file with cutplane information
  o_json << "[" << endl;
  for(int i = 0; i < cpl.size(); i++)
    {
    auto &cp = cpl[i];
    o_json << "  {" << endl;
    o_json << "    \"theta\": " << cp.second << "," << endl;
    o_json << "    \"r\": " << cp.first << endl;
    o_json << "  }" << (i+1 < (int) cpl.size() ? "," : "") << endl;
    }
  o_json << "]" << endl;

  // Generate a PNG for cutting on paper
  cout << "  Generating PNG template " << endl;
  api.Execute(
        "-clear -push carve1 -swapdim PRS -extrude-seg -swapdim LPI -thresh 0 inf 1 0 "
        "-dilate 1 5x0x5 -dilate 0 5x0x5 -dup -dilate 0 2x0x2 -subtract -shift 1 "
        "-push cuts -thresh 0 inf 1 0 -min -popas F -clear "
        "-push slabvol -replace 0 255 -split "
        "-foreach -thresh 0 0 1 0 -swapdim PRS -extrude-seg -swapdim ARS -extrude-seg -swapdim LPI -thresh 0 0 1 0 -endfor "
        "-scale 0.1 -merge -int 0 -insert F 1 -background 255 -reslice-identity -push F -replace 1 255 -min "
        "-dup -push dots_slab -thresh 0 0 1 0 -swapdim PRS -extrude-seg -swapdim ARS -extrude-seg -swapdim LPI "
        "-thresh 0 0 0 255 -background 0 -reslice-identity -as dd -min -push dd -thresh 255 255 1 0 -dilate 0 2x0x2 -times "
        "-slice y 50%% -as png_slice -clear");

  // Generate replace commands for R/G/B (like -oli but with in-memory palette)
  for(unsigned int c = 0; c < 3; c++)
    {
    ostringstream oss;
    oss << "-replace ";
    for(unsigned int i = 0; i < 64; i++)
      {
      int val = (palette[i] >> (2-c) * 8) & 0xff;
      oss << (i+1) << " " << val << " ";
      }
    api.Execute("-push png_slice %s", oss.str().c_str());
    }
  string fn_png = get_output_filename(param, SLAB_CUT_PNG, slab_id);
  api.Execute("-type uchar -foreach -flip y -endfor -omc %s", fn_png.c_str());

  // Get statistics on the blocks
  auto stat_vol = get_label_stats(api.GetImage("slabvol"));

  // Get statistics on the dots. For each dot
  auto stat_dots = get_label_stats(api.GetImage("dots_slab"));
  for(auto it : stat_dots)
    {
    auto &ctr = it.second.center_of_mass;
    int label = get_block_label(ctr[0], ctr[2], cpl);
    cout << "Dot " << it.second.label << " Center " << ctr << " Label " << label << endl;
    }

  // Generate the tex file
  ImagePointer slice = api.GetImage("png_slice");
  double ext_x = slice->GetBufferedRegion().GetSize()[0] * slice->GetSpacing()[0];
  double ext_y = slice->GetBufferedRegion().GetSize()[1] * slice->GetSpacing()[1];

  ofstream otex(get_output_filename(param, SLAB_CUT_TEXLET, slab_id).c_str());

  // Generate Latex. Here it is easier to use fprintf
  FILE *ftex = fopen(get_output_filename(param, SLAB_CUT_TEXLET, slab_id).c_str(), "wt");
  fprintf(ftex,
          "\\begin{figure}\n"
          "  \\centering\n"
          "  \\scalebox{-1}[1]{\\includegraphics[width=%fmm,height=%fmm]{%s}}\n"
          "  \\hfill"
          "  \\scalebox{1}[1]{\\includegraphics[width=%fmm,height=%fmm]{%s}}\n"
          "  \\newline \\newline \\newline \n",
          ext_x, ext_y, itksys::SystemTools::GetFilenameWithoutExtension(fn_png).c_str(),
          ext_x, ext_y, itksys::SystemTools::GetFilenameWithoutExtension(fn_png).c_str());

  // Insert a table with available labels
  fprintf(ftex, "  \\begin{tabular}{|c|p{3in}}\n");
  fprintf(ftex, "    \\hline\n");
  for(auto it : stat_vol)
    {
    // Print the label
    fprintf(ftex, "    \\cellcolor[rgb]{%f,%f,%f}\\textbf{Block %02d-%d} & \n",
            ((palette[it.first-1] >> 16) & 0xff) / 255.0,
            ((palette[it.first-1] >> 8) & 0xff) / 255.0,
            ((palette[it.first-1] >> 0) & 0xff) / 255.0,
            slab_id, it.first);

    // Report the dots
    for(auto itd : stat_dots)
      {
      auto &ctr = itd.second.center_of_mass;
      int label = get_block_label(ctr[0], ctr[2], cpl);
      if(label == it.first)
        {
        fprintf(ftex, "    \\small Dot %d at depth $%4.2f$mm of $%4.2f$mm \\newline \n",
                itd.second.label, ctr[1] - slab.y0, slab.y1 - slab.y0);
        }
      }

    // End the table entry
    fprintf(ftex, "    \\\\\n");
    fprintf(ftex, "    \\hline\n");
    }
  fprintf(ftex, "  \\end{tabular}\n");
  fprintf(ftex, "  \\caption{\\textbf{Specimen %s slab %02d}}\n", param.id_string.c_str(), slab_id);
  fprintf(ftex, "\\end{figure}\n");
  fclose(ftex);
}


int main(int argc, char *argv[])
{
  itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(1);

  if(argc < 2)
    return usage();

  Parameters param;

  for(int i = 1; i < argc; ++i)
    {
    string arg = argv[i];
    unsigned int n_trail = argc - (i+1);
    
    if(arg == "-s" && n_trail > 0)
      {
      param.id_string = argv[++i];
      }
    else if(arg == "-h" && n_trail > 0)
      {
      string side = argv[++i];
      if (side == "L" || side == "l")
        param.side = 'L';
      else if (side == "R" || side == "r")
        param.side = 'R';
      else
        {
        printf("Side must be L or R\n");
        return -1;
        }
      }
    else if(arg == "-i" && n_trail > 1)
      {
      param.fn_input_mri = argv[++i];
      param.fn_input_seg = argv[++i];
      }
    else if(arg == "-d")
      {
      param.fn_input_dots = argv[++i];
      }
    else if(arg == "-C")
      {
      param.flag_no_cuts = true;
      }
    else if(arg == "-a")
      {
      param.fn_input_avoidance = argv[++i];
      }
    else if(arg == "-o" && n_trail > 0)
      {
      param.fn_output_root = argv[++i];
      }
    else if(arg == "--no-hemi-mold")
      {
      param.flag_print_hemisphere_mold = false;
      }
    else if(arg == "--slab")
      {
      param.selected_slab = atoi(argv[++i]);
      }
    else 
      {
      printf("Command %s is unknown or too few parameters provided\n", argv[i]);
      return -1;
      }
    }

  // C3D output should be piped to a log file
  string fn_c3d_log = get_output_filename(param, HEMI_C3D_LOG);
  ofstream c3d_log(fn_c3d_log.c_str());

  // Use c3d to generate a mold
  try 
    { 
    ConvertAPIType api;
    api.RedirectOutput(c3d_log, c3d_log);

    // Read the mask image and dots, avoidance images
    api.Execute("-verbose %s -popas mask", param.fn_input_seg.c_str());

    if(param.fn_input_dots.size())
      // TODO: if we want to widen the dots to 3x3x3, add this code instead:
      // "%s -swapdim LPI -verbose -dup -scale 0 -popas ref -replace 0 255 -split -popas BG -foreach -dilate 1 1x1x1 -trim-to-size 3x3x3vox -info -insert ref 1 -reslice-identity -endfor -push BG -merge -replace 255 0 -popas dots"
      api.Execute("%s -swapdim LPI -popas dots", param.fn_input_dots.c_str());
    else
      api.Execute("-push mask -scale 0 -swapdim LPI -popas dots");

    if(param.fn_input_avoidance.size())
      api.Execute("%s -swapdim LPI -popas avoid", param.fn_input_avoidance.c_str());
    else
      api.Execute("-push mask -scale 0 -swapdim LPI -popas avoid");

    ImagePointer i_mask = api.GetImage("mask");
    ImagePointer i_dots = api.GetImage("dots");
    ImagePointer i_avoid = api.GetImage("avoid");

    // The trim amount should be at least equal to floor thickness and wall thickness
    double trim_radius = std::max(param.mold_wall_thickness_mm,
                                  std::max(param.mold_floor_thickness_mm, 5.));

    // The first thing is to reorient the image, upsample to desired resolution, 
    // and extract the largest connected component
    //
    // We will reorient the image to LPI orientation so that the direction of the 
    // voxel (x,y,z) axes and physical (x,y,z) axes coincide.
    //
    // In this frame, the slits are orthogonal to the y axis (anterior-posterior)
    //
    // The bottom of the mold is either upper (R) or lower (L) range of the x-axis
    // (should be specified as input)
    cout << "Extracting hemisphere main connected component and resampling" << endl;
    api.Execute(
          "-push mask -swapdim LPI -thresh 1 1 1 0 -as comp_raw "
          "-dilate 1 %dx%dx%d -dilate 0 %dx%dx%d -as comp "
          "-trim %fmm -resample-mm %fmm -as H ",
          param.preproc_dilation, param.preproc_dilation, param.preproc_dilation,
          param.preproc_erosion, param.preproc_erosion, param.preproc_erosion,
          trim_radius, param.mold_resolution_mm);

    // The image H is the trimmed and resampled hemisphere. We will need its dimensions
    // for subsequent work
    ImageType *i_comp_raw = api.GetImage("comp_raw");
    ImageType *i_comp_resampled = api.GetImage("H");
    auto sz = i_comp_resampled->GetBufferedRegion().GetSize();
    auto sp = i_comp_resampled->GetSpacing();
    auto org = i_comp_resampled->GetOrigin();

    // The length of the brain that will be slabbed
    double len_y = sz[1]*sp[1] - 2 * trim_radius;
    double off_y = (sz[1]*sp[1]/2 - 5);
    double off_x = 5;

    // Things that depend on hemisphere side are the direction of extrusion and
    // where we start the base
    const char *ext_dir_1 = param.side == 'R' ? "RPS" : "LPS";
    const char *ext_dir_2 = param.side == 'R' ? "LPS" : "RPS";

    // Compute the slab positions (TODO: optimize)
    itk::ContinuousIndex<double, 3> idxCenter;
    itk::Point<double, 3> pCenter;
    for(unsigned int i = 0; i < 3; i++)
      idxCenter[i] = sz[i] * 0.5;
    i_comp_resampled->TransformContinuousIndexToPhysicalPoint(idxCenter, pCenter);
    double y_center = -pCenter[1];

    // The number of slabs
    int n_slabs = (int) ceil(off_y / param.slit_spacing_mm) * 2;
    printf("Slabs %d, center slab coordinates: %f\n", n_slabs, y_center);

    // Create the slabs
    std::vector<Slab> slabs(n_slabs);

    // The slabs should go from anterior to posterior, so from increasing y
    // coordinate to decreasing y coordinate
    for(int i = 0; i < n_slabs; i++)
      {
      slabs[i].y1 = y_center + (n_slabs/2. - i) * param.slit_spacing_mm;
      slabs[i].y0 = slabs[i].y1 - param.slit_spacing_mm;
      }

    // The next step is printing the main mold, which user may omit
    if(param.flag_print_hemisphere_mold)
      {
      cout << "Generating hemisphere mold" << endl;

      // For each slab, generate a slit
      api.Execute("-clear -push H -cmp -popas z -popas y -popas x "
                  "-push H -scale 0 -shift 4");

      for(int i = 0; i <= n_slabs; i++)
        {
        // Cutting plane depth
        double y_cut = (i < n_slabs) ? slabs[i].y0 : slabs[i-1].y1;

        // Apply cutting plane
        api.Execute("-push y -shift %f -abs -stretch 0 %f -4 4 -clip -4 4 -min",
                    -y_cut, param.slit_width_mm);
        }
      api.Execute("-as res1 -o /tmp/res1.nii.gz");

      // The next step is to create the base. The base should begin mold_floor_thickness
      // away from the hemisphere segmentation. Here, compute offset in voxels
      double floor_offset_vox = (trim_radius - param.mold_floor_thickness_mm) / param.mold_resolution_mm;

      // Create a solid base
      api.Execute(
            "-clear -push H -swapdim %s -cmv -pick 0 "
            "-stretch %f %f 4 -4 -clip -4 4 -swapdim LPI -as base",
            ext_dir_1, floor_offset_vox - 1.0, floor_offset_vox + 1.0);

      // Create solid ends around the center
      api.Execute("-clear -push H -push base -push res1 -max "
                  "-copy-transform -pad 5x5x5 5x5x5 -4 -as mold -o /tmp/mold.nii.gz ");
      /*
      api.Execute(
        "-push x -info -thresh %f inf 4 -4 -max "
        "-push y -info -thresh %f %f -4 4 -max "
        "-insert H 1 -copy-transform "
        "-pad 5x5x5 5x5x5 -4 -as mold ",
        off_x, y_center-off_y, y_center+off_y);
        */

      // Extrude the brain image and trim extra plastic
      int p3 = (int) ceil(param.mold_wall_thickness_mm / param.mold_resolution_mm);
      api.Execute(
        "-clear -push comp -thresh -inf 0.5 4 -4 "
        "-swapdim %s -extrude-seg -swapdim LPI -dup "
        "-swapdim %s -extrude-seg "
        "-thresh -inf 0 0 1 -dilate 0 0x%dx%d -stretch 0 1 4 -4 -swapdim LPI -min "
        "-insert mold 1 -background 4 -reslice-identity "
        "-push mold -min -as carved "
        "-thresh 0 inf 1 0 -o /tmp/b.nii.gz", ext_dir_1, ext_dir_2, p3, p3);

      // Get the connected component image and mold
      ImagePointer i_comp = api.GetImage("comp");
      ImagePointer i_mold = api.GetImage("carved");

      save_image(i_comp.GetPointer(), get_output_filename(param, HEMI_VOLUME_IMAGE));
      save_image(i_mold.GetPointer(), get_output_filename(param, HEMI_MOLD_IMAGE));
      }

    // Get the coordinates of the cutting lines used above. In the future
    // we may want to optimize over these instead. 
    //
    // Process a slab
    for(int i = 0; i < n_slabs; i++)
      {
      if(param.selected_slab < 0 || param.selected_slab == i)
        {
        cout << "Generating mold for slab " << i << " of " << n_slabs << endl;
        cout << "  Y range: " << slabs[i].y0 << ", " << slabs[i].y1 << endl;
        if(param.flag_no_cuts)
          process_slab_nocuts(param, i, slabs[i], i_comp_raw, i_comp_resampled, i_dots, i_avoid);
        else
          process_slab(param, i, slabs[i], i_comp_raw, i_comp_resampled, i_dots, i_avoid);
        }
      }

    // Generate the final tex file
    FILE *f_tex = fopen(get_output_filename(param, GLOBAL_TEXFILE).c_str(), "wt");
    fprintf(f_tex,
            "\\documentclass{article}\n"
            "\\usepackage[table]{xcolor}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage[letterpaper, margin=1in]{geometry}\n"
            "\\renewcommand{\\arraystretch}{1.5}\n"
            "\\setlength\\tabcolsep{0.25in}\n"
            "\\begin{document}\n");

    for(int i = 0; i < n_slabs; i++)
      {
      string fn_tex = get_output_filename(param, SLAB_CUT_TEXLET, i);
      if(itksys::SystemTools::FileExists(fn_tex))
        fprintf(f_tex, "  \\include{%s}\n",
                itksys::SystemTools::GetFilenameWithoutExtension(fn_tex).c_str());
      }
    fprintf(f_tex, "\\end{document}\n");
    fclose(f_tex);
    }
  catch(ConvertAPIException &exc)
    {
    cerr << "ConvertAPIException caught: " << exc.what() << endl;
    return -1;
    }

  return 0;
}
