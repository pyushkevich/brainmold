#include "ConvertAPI.h"
#include <string>
#include <cstdio>
#include <cmath>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include "SparseMatrix.h"
#include "GentleNLP.h"
#include "IPOptProblemInterface.h"
#include <IpIpoptApplication.hpp>
#include <itksys/SystemTools.hxx>

#include <vnl/vnl_cost_function.h>
#include <vnl/algo/vnl_amoeba.h>

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
  string fn_output_root;
  string id_string;

  // Parameters for mold creation
  double slit_spacing_mm = 10.0;
  double slit_width_mm = 0.8;
  double mold_resolution_mm = 0.4;
  double mold_wall_thickness_mm = 3;
  double mold_floor_thickness_mm = 3;

  // The amount of dilation and erosion applied to the mask as a
  // preprocessing step
  int preproc_dilation = 5, preproc_erosion = 3;

  // Flags
  bool flag_print_hemisphere_mold = true;

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
  SLAB_C3D_LOG,
  SLAB_OPTIMIZATION_LOG,
  SLAB_CUTPLANE_JSON,
  SLAB_CUT_PNG
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
      sprintf(fn_out, "%s/%s_hemi_mold.nii.gz", root, id);
      break;
    case HEMI_VOLUME_IMAGE:
      sprintf(fn_out, "%s/%s_hemi_volume.nii.gz", root, id);
      break;
    case HEMI_C3D_LOG:
      sprintf(fn_out, "%s/logs/%s_hemi_c3d_log.txt", root, id);
      break;
    case SLAB_MOLD_IMAGE:
      sprintf(fn_out, "%s/%s_slab%02d_mold.nii.gz", root, id, slab);
      break;
    case SLAB_VOLUME_IMAGE:
      sprintf(fn_out, "%s/%s_slab%02d_volume.nii.gz", root, id, slab);
      break;
    case SLAB_C3D_LOG:
      sprintf(fn_out, "%s/logs/%s_slab%02d_c3d_log.txt", root, id, slab);
      break;
    case SLAB_OPTIMIZATION_LOG:
      sprintf(fn_out, "%s/logs/%s_slab%02d_opt_log.txt", root, id, slab);
      break;
    case SLAB_CUTPLANE_JSON:
      sprintf(fn_out, "%s/%s_slab%02d_cutplanes.json", root, id, slab);
      break;
    case SLAB_CUT_PNG:
      sprintf(fn_out, "%s/%s_slab%02d_cuts.png", root, id, slab);
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
    "  -i <mri> <mask>          Input MRI and brain mask\n"
    "  -o <dir>                 Output directory\n"
    "optional inputs:\n"
    "  -d <image>               Dots file\n"
    "  --no-hemi-mold           Skip hemisphere mold printing\n"
    "  --slab <index>           Only generate for one slab\n"
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

/**
 * Exponentiation
 */
class ExpOperatorTraits
{
public:

  static double Operate(double a)
  {
    return exp(a);
  }

  static gnlp::Expression *Differentiate(
      gnlp::Problem *p, gnlp::Expression *self,
      gnlp::Expression *, gnlp::Expression *dA)
  {
    return gnlp::MakeProduct(p, self, dA);
  }

  static std::string GetName(gnlp::Expression *a)
  {
    std::ostringstream oss;
    oss << "exp(" << a->GetName() << ")";
    return oss.str();
  }
};


/**
 * Sigmoid operation
 */
class SigmoidOperatorTraits
{
public:

  static double Operate(double a)
  {
    double ea = exp(-a);
    return 1.0 / (1.0 + ea);
  }

  static gnlp::Expression *Differentiate(
      gnlp::Problem *p, gnlp::Expression *self,
      gnlp::Expression *a, gnlp::Expression *dA)
  {
    auto *exp_minus_a = new gnlp::UnaryExpression<ExpOperatorTraits>(p, new gnlp::Negation(p, a));
    return gnlp::MakeProduct(p, self, self, gnlp::MakeProduct(p, dA, exp_minus_a));
  }

  static std::string GetName(gnlp::Expression *a)
  {
    std::ostringstream oss;
    oss << "sigmoid(" << a->GetName() << ")";
    return oss.str();
  }
};


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


#include <queue>
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


class SlabCutPlaneBruteOptimization
{
public:
  typedef ImmutableSparseMatrix<double> SparseMatrixType;

  /** Initialize graph data structures for an input slab */
  SlabCutPlaneBruteOptimization(ImageType *slab)
  {
    // Output goes to std::out
    m_SOut = &std::cout;

    // On the first pass, label the image with vertex indices and set coord arrays
    typedef itk::Image<int, 3> LabelImage;
    typename LabelImage::Pointer imgLabel = LabelImage::New();
    imgLabel->CopyInformation(slab);
    imgLabel->SetRegions(slab->GetBufferedRegion());
    imgLabel->Allocate();
    imgLabel->FillBuffer(-1);

    auto region = slab->GetBufferedRegion();
    itk::ImageRegionConstIteratorWithIndex<ImageType> itSrc(slab, region);
    itk::ImageRegionIterator<LabelImage> itDest(imgLabel, region);
    unsigned int n = 0;
    for( ; !itSrc.IsAtEnd(); ++itSrc, ++itDest)
      {
      if(itSrc.Get() > 0)
        itDest.Set(n++);
      }

    // Second pass, generate coordinate arrays
    m_X.set_size(n, 3);
    itk::ImageRegionConstIteratorWithIndex<ImageType> itSrc2(slab, region);
    int k = 0;
    for(; !itSrc2.IsAtEnd(); ++itSrc2)
      {
      if(itSrc2.Get())
        {
        // Map index to RAS coordinate
        auto idx = itSrc2.GetIndex();
        itk::Point<double, 3> pt;
        slab->TransformIndexToPhysicalPoint(idx, pt);
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

    // Compute connected components in the input dataset
    m_WholeComp = count_connected_components_bfs(&m_S, m_NodeLabel.data_block(), m_CompIndex.data_block());
  }

  /** Set up optimization for specific number of cut planes */
  void SetupOptimization(int extra_cuts_w, int extra_cuts_h)
  {
    // Figure out the minimum number of cuts (based on slide dimensions)
    m_NumCutsW = (int) (m_XDim[0] / 70);
    m_NumCutsH = (int) (m_XDim[2] / 45);

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
      double r_i = x[i+1] + cos(theta) * m_XCtr[0] + cos(theta) * m_XCtr[2];
      double theta_i = theta;
      cpl.push_back(std::make_pair(r_i, theta_i));
      }

    // Vertical cuts have opposite orientation as theta
    for(unsigned int i = m_NumCutsW; i < m_NumCutsH+m_NumCutsW; i++)
      {
      // Adjusting the R for the origin
      double r_i = x[i+1] - sin(theta) * m_XCtr[0] + cos(theta) * m_XCtr[2];
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
    double cos_phi = cos(phi), sin_phi = sin(phi);

    *m_SOut << "==== ITERATION " << setw(3) << ++m_Iter << " ====" << endl;
    *m_SOut << "  Parameters: " << vnl_vector<double>(x, m_NumUnknowns) << endl;

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
        }
      }

    // Now compute statistics for each cut (extents, volume, etc);
    unsigned int nr = 1 << cpl.size();
    vnl_matrix<double> extents(nr, 4, 0);
    vnl_vector<double> volumes(nr, 0.);
    for(unsigned int i = 0; i < m_X.rows(); i++)
      {
      int l = m_NodeLabel[i];
      double u = m_X[i][0] * cos_phi + m_X[i][2] * sin_phi;
      double v = -m_X[i][0] * sin_phi + m_X[i][2] * cos_phi;
      if(volumes[l] == 0 || u < extents(l, 0)) extents(l, 0) = u;
      if(volumes[l] == 0 || u > extents(l, 1)) extents(l, 1) = u;
      if(volumes[l] == 0 || v < extents(l, 2)) extents(l, 2) = v;
      if(volumes[l] == 0 || v > extents(l, 3)) extents(l, 3) = v;
      volumes[l]++;
      }

    // Normalize the volumes
    volumes *= 1.0 / volumes.sum();

    // Compute penalty terms
    double penalty_extent = 0;
    double sum_vol = 0, ssq_vol = 0;
    int n_vol = 0;
    for(unsigned int l = 0; l < nr; l++)
      {
      if(volumes[l] > 0)
        {
        double w = extents(l,1) - extents(l,0);
        double h = extents(l,3) - extents(l,2);
        double ext_min = min(w, h), ext_max = max(w, h);

        char buffer[256];
        sprintf(buffer, "  Piece %d:  Extent: %4.2f by %4.2f  Volume = %6.4f", l, ext_max, ext_min, volumes[l]);
        *m_SOut << buffer<< endl;

        if(ext_max > 74.)
          penalty_extent += 100000 * (ext_max - 74.);
        if(ext_min > 49.)
          penalty_extent += 100000 * (ext_min - 49.);
        sum_vol += volumes[l];
        ssq_vol += volumes[l] * volumes[l];
        n_vol++;
        }
      }
    *m_SOut << "  Extent violation penalty: " << penalty_extent << endl;


    // Compute number of connected components
    int n_comp = count_connected_components_bfs(&m_S, m_NodeLabel.data_block(), m_CompIndex.data_block());
    double penalty_conn = n_comp * 1.0;

    *m_SOut << "  Connected components in graph: " << n_comp << endl;
    *m_SOut << "  Connected components penalty: " << penalty_conn << endl;

    // Compute the variance in volume
    double var_vol = (ssq_vol - sum_vol / n_vol) / n_vol;
    double std_vol = sqrt(var_vol);
    *m_SOut << "  Volume variance: " << std_vol << endl;
    *m_SOut << "  Volume variance penalty:" << std_vol << endl;

    // Total objective
    double total_obj = penalty_extent + penalty_conn + var_vol;
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
      ip[i+1] = (i+1) * m_XDim[0] / (m_NumCutsW+1.) - m_XDim[0] / 2.0;
      }

    for(int i = 0; i < m_NumCutsH; i++)
      {
      ip[i+m_NumCutsW+1] = (i+1) * m_XDim[2] / (m_NumCutsH+1.) - m_XDim[2] / 2.0;
      }

    return ip;
  }

  void RedirectOutput(std::ostream &sout) { m_SOut = &sout; }

  int GetNumberOfComponentsInSlab() const { return m_WholeComp; }

private:
  // Sparse matrix representation of non-zero voxels
  SparseMatrixType m_S;

  // Matrix of xyz coordinates of the foreground voxels in the mask
  vnl_matrix<double> m_X;

  // Center of mass
  vnl_vector_fixed<double, 3> m_XMin, m_XMax, m_XDim, m_XCtr;

  // Node labels and connected component labels
  vnl_vector<int> m_NodeLabel, m_CompIndex;

  // Number of cuts for current optimization setup
  int m_NumCutsW, m_NumCutsH, m_NumUnknowns, m_Iter, m_WholeComp;

  // Output stream
  std::ostream *m_SOut;
};

// A nice palette generated by https://mokole.com/palette.html
int palette[] = {
  0xa9a9a9, 0xdcdcdc, 0x2f4f4f, 0x556b2f, 0x6b8e23, 0xa0522d, 0x228b22, 0x7f0000,
  0x191970, 0x708090, 0x483d8b, 0x5f9ea0, 0x3cb371, 0xbc8f8f, 0x663399, 0xbdb76b,
  0xcd853f, 0x4682b4, 0x000080, 0xd2691e, 0x9acd32, 0xcd5c5c, 0x32cd32, 0xdaa520,
  0x7f007f, 0x8fbc8f, 0xb03060, 0xd2b48c, 0x66cdaa, 0xff0000, 0x00ced1, 0xffa500,
  0xffd700, 0xc71585, 0x0000cd, 0x7cfc00, 0x00ff00, 0xba55d3, 0x8a2be2, 0x00ff7f,
  0x4169e1, 0xe9967a, 0xdc143c, 0x00ffff, 0x00bfff, 0x9370db, 0x0000ff, 0xff6347,
  0xd8bfd8, 0xff00ff, 0xdb7093, 0xf0e68c, 0xffff54, 0x6495ed, 0xdda0dd, 0x87ceeb,
  0xff1493, 0xafeeee, 0xee82ee, 0x98fb98, 0x7fffd4, 0xff69b4, 0xffe4c4, 0xffb6c1
};


class SlabCutPlaneOptimization
{
public:
  typedef ImmutableSparseMatrix<double> SparseMatrixType;

  SlabCutPlaneOptimization(ImageType *slab) : m_Problem(NULL)
  {
    // On the first pass, label the image with vertex indices and set coord arrays
    typedef itk::Image<int, 3> LabelImage;
    typename LabelImage::Pointer imgLabel = LabelImage::New();
    imgLabel->CopyInformation(slab);
    imgLabel->SetRegions(slab->GetBufferedRegion());
    imgLabel->Allocate();
    imgLabel->FillBuffer(-1);

    auto region = slab->GetBufferedRegion();
    itk::ImageRegionConstIteratorWithIndex<ImageType> itSrc(slab, region);
    itk::ImageRegionIterator<LabelImage> itDest(imgLabel, region);
    unsigned int n = 0;
    for( ; !itSrc.IsAtEnd(); ++itSrc, ++itDest)
      {
      if(itSrc.Get())
        itDest.Set(n++);
      }

    // Second pass, generate coordinate arrays
    m_X.set_size(n, 3);
    itk::ImageRegionConstIteratorWithIndex<ImageType> itSrc2(slab, region);
    int k = 0;
    for(; !itSrc2.IsAtEnd(); ++itSrc2)
      {
      if(itSrc2.Get())
        {
        // Map index to RAS coordinate
        auto idx = itSrc2.GetIndex();
        itk::Point<double, 3> pt;
        slab->TransformIndexToPhysicalPoint(idx, pt);
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
  }

  // Set up the optimization with given number of horizontal and vertical cuts
  void SetupOptimization(unsigned int n_horiz, unsigned int n_vert)
  {
    // Create a new optimization problem
    if(m_Problem)
      delete m_Problem;
    m_Problem = new gnlp::ConstrainedNonLinearProblem();

    // Set up the orientation of the stack. For simplicity, we define two variables
    // for cosine and sine of the angle, that have to add up to one
    v_CosTheta = m_Problem->AddVariable("CosTheta", 1, sqrt(2.) / 2., 1.0);
    v_SinTheta = m_Problem->AddVariable("SinTheta", 0);

    // Add constraint on cosine and sine
    m_Problem->AddConstraint(new gnlp::BinarySum(
                               m_Problem,
                               new gnlp::Square(m_Problem, v_CosTheta),
                               new gnlp::Square(m_Problem, v_SinTheta)),
                             "sincos", 1.0, 1.0);

    // Create expressions for the sum of heights and widths
    gnlp::BigSum *sum_width = new gnlp::BigSum(m_Problem);
    gnlp::BigSum *sum_height = new gnlp::BigSum(m_Problem);

    // Expressions for the cut planes in polar format
    x_CutPlaneR.clear();
    x_CutPlaneCosTheta.clear();
    x_CutPlaneSinTheta.clear();

    // Total width and height of dataset
    double w = m_XMax[0] - m_XMin[0], h = m_XMax[2] - m_XMin[2];

    // For each plane, set up the variables
    v_CutWidths.clear();
    gnlp::Expression *r_prev = new gnlp::Constant(m_Problem, m_XMin[0]);
    for(unsigned int i = 0; i <= n_horiz; i++)
      {
      v_CutWidths.push_back(m_Problem->AddVariable("W_i", w / (n_horiz+1), 0, 75));
      sum_width->AddSummand(v_CutWidths.back());
      if(i < n_horiz)
        {
        x_CutPlaneR.push_back(new gnlp::BinarySum(m_Problem, r_prev, v_CutWidths.back()));
        x_CutPlaneCosTheta.push_back(v_CosTheta);
        x_CutPlaneSinTheta.push_back(v_SinTheta);
        r_prev = x_CutPlaneR.back();
        }
      }

    v_CutHeights.clear();
    r_prev = new gnlp::Constant(m_Problem, m_XMin[2]);
    for(unsigned int i = 0; i <= n_vert; i++)
      {
      v_CutHeights.push_back(m_Problem->AddVariable("H_i", h / (n_vert+1), 0, 50));
      sum_height->AddSummand(v_CutHeights.back());
      if(i < n_vert)
        {
        x_CutPlaneR.push_back(new gnlp::BinarySum(m_Problem, r_prev, v_CutHeights.back()));
        x_CutPlaneCosTheta.push_back(new gnlp::Negation(m_Problem, v_SinTheta));
        x_CutPlaneSinTheta.push_back(v_CosTheta);
        r_prev = x_CutPlaneR.back();
        }
      }

    // Add a constraint that the heights must add up to total width and height
    m_Problem->AddConstraint(sum_width, "sum_w", w, w);
    m_Problem->AddConstraint(sum_height, "sum_h", h, h);

    cout << "Sum Height: " << sum_height->GetName() << endl;

    // Compute the volume of each partition. This is obtained by applying a sigmoid
    // function to the inputs with respect to each plane
    gnlp::VarVecArray alpha(m_X.rows(), gnlp::VarVec(x_CutPlaneR.size()));
    gnlp::VarVecArray beta(m_X.rows(), gnlp::VarVec(x_CutPlaneR.size()));

    // Number of regions
    unsigned int n_rgn = (1 << x_CutPlaneR.size());

    // Number of pixels for each slide
    std::vector<gnlp::BigSum *> rgn_volume(n_rgn, NULL);
    for(unsigned int k = 0; k < n_rgn; k++)
      rgn_volume[k] = new gnlp::BigSum(m_Problem);

    for(unsigned int j = 0; j < m_X.rows(); j++)
      {
      // Coordinates of the point
      gnlp::Expression *xj = new gnlp::Constant(m_Problem, m_X[j][0]);
      gnlp::Expression *yj = new gnlp::Constant(m_Problem, m_X[j][2]);

      // Compute the membership of the point with respect to each cut plane
      for(unsigned int i = 0; i < x_CutPlaneR.size(); i++)
        {
        // Projection on the normal vector of the cut
        gnlp::Expression *dij = new gnlp::BinarySum(
                                  m_Problem,
                                  new gnlp::BinaryProduct(m_Problem, xj, x_CutPlaneCosTheta[i]),
                                  new gnlp::BinaryProduct(m_Problem, yj, x_CutPlaneSinTheta[i]));

        // A value whose sign indicates what side of the plane we are on
        gnlp::Expression* dr = new gnlp::BinaryDifference(m_Problem, dij, x_CutPlaneR[i]);

        // Apply the sigmoid here
        typedef gnlp::UnaryExpression<SigmoidOperatorTraits> Sigmoid;
        alpha[j][i] = new Sigmoid(m_Problem, dr);
        beta[j][i] = new gnlp::BinaryDifference(
                       m_Problem, new gnlp::Constant(m_Problem, 1.0), alpha[j][i]);
        }

      // Membership is product of alphas and betas
      for(unsigned int k = 0; k < n_rgn; k++)
        {
        gnlp::Expression *ab_prod = NULL;
        for(unsigned int i = 0; i < x_CutPlaneR.size(); i++)
          {
          bool use_alpha = k & (1 << i);
          gnlp::Expression *ab = use_alpha ? alpha[j][i] : beta[j][i];
          ab_prod = ab_prod ? gnlp::MakeProduct(m_Problem, ab_prod, ab) : ab;
          }

          // Add this product to the volume for k
          rgn_volume[k]->AddSummand(ab_prod);
        }
      }

    // Finally, we want to compute the variance in volume
    gnlp::BigSum *sum_vol = new gnlp::BigSum(m_Problem);
    gnlp::BigSum *sum_vol_sq = new gnlp::BigSum(m_Problem);
    for(unsigned int k = 0; k < n_rgn; k++)
      {
      sum_vol->AddSummand(rgn_volume[k]);
      sum_vol_sq->AddSummand(new gnlp::Square(m_Problem, rgn_volume[k]));
      }

    // This is the variance in volume
    gnlp::Expression *var_vol =
        new gnlp::ScalarProduct(
          m_Problem,
          new gnlp::BinaryDifference(
            m_Problem,
            sum_vol_sq,
            new gnlp::ScalarProduct(
              m_Problem,
              new gnlp::Square(m_Problem, sum_vol),
              1.0 / n_rgn)),
          1.0 / n_rgn);

    // Add this to the objective
    m_Problem->SetObjective(var_vol);
    m_Problem->SetupProblem(false, false);
  }

  // Test sigmoid
  void TestSigmoid()
  {
    auto *p = new gnlp::ConstrainedNonLinearProblem();
    auto *v = p->AddVariable("x", -4.3);
    auto *s = new gnlp::UnaryExpression<SigmoidOperatorTraits>(p, v);
    p->SetObjective(s);
    p->SetupProblem(false, true);
    printf("Sigmoid test done\n");
  }

  // Perform optimization
  CutPlaneList RunOptimization()
  {
    FILE *f_con = fopen("/tmp/f_con.log","wt");
    SmartPtr<IPOptProblemInterface> ip_opt = new IPOptProblemInterface(m_Problem, false);
    ip_opt->log_constraints(f_con);
    IpoptApplication *app = IpoptApplicationFactory();
    app->Options()->SetNumericValue("tol", 1e-8);
    app->Options()->SetStringValue("linear_solver", "ma57");
    app->Options()->SetIntegerValue("max_iter", 200);
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");

    auto status = app->Initialize();
    if(status != Solve_Succeeded)
      throw ConvertAPIException("IPOpt Initialization Failed");

    status = app->OptimizeTNLP(GetRawPtr(ip_opt));

    // Construct the solution in the form of a list of cut planes
    CutPlaneList cpl;
    for(unsigned int i = 0; i < x_CutPlaneR.size(); i++)
      {
      CutPlane cp = make_pair(
                      x_CutPlaneR[i]->Evaluate(),
                      atan2(x_CutPlaneSinTheta[i]->Evaluate(), x_CutPlaneCosTheta[i]->Evaluate()));
      cpl.push_back(cp);
      }

    return cpl;
  }

protected:

  // Sparse matrix representation of non-zero voxels
  SparseMatrixType m_S;

  // Matrix of xyz coordinates of the foreground voxels in the mask
  vnl_matrix<double> m_X;

  // Center of mass
  vnl_vector_fixed<double, 3> m_XMin, m_XMax;

  // Node labels
  vnl_vector<int> m_NodeLabel;

  // The problem
  gnlp::ConstrainedNonLinearProblem *m_Problem;

  // Horizontal and vertical cut plane data
  std::vector<gnlp::Variable *> v_CutHeights, v_CutWidths;

  // Cutplane properties
  std::vector<gnlp::Expression *> x_CutPlaneR, x_CutPlaneSinTheta, x_CutPlaneCosTheta;

  // Orientation of the cuts
  gnlp::Variable *v_CosTheta, *v_SinTheta;
};


void process_slab(Parameters &param, int slab_id, Slab &slab, ImageType *i_hemi_raw, ImageType *i_hemi_mask)
{
  // C3D output should be piped to a log file
  ofstream c3d_log(get_output_filename(param, SLAB_C3D_LOG, slab_id).c_str());
  ofstream o_json(get_output_filename(param, SLAB_CUTPLANE_JSON, slab_id));

  // Create our own API
  ConvertAPIType api;
  api.RedirectOutput(c3d_log, c3d_log);

  // Deal with the raw image (for visualization)
  api.AddImage("raw", i_hemi_raw);
  api.Execute("-verbose -push raw -cmp -pick 1 -thresh %f %f 1 0 "
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
  api.Execute("-dilate 0 11x0x11 -as slab_trim -o /tmp/slab_trim.nii.gz");

  // Cut plane list
  CutPlaneList cpl;

  // IPOpt based optimization
  if(true)
    {
    // Optimization output should be piped to a log file
    string fn_opt_log = get_output_filename(param, SLAB_OPTIMIZATION_LOG, slab_id);
    ofstream opt_log(fn_opt_log.c_str());

    // Set up the optimizer
    SlabCutPlaneBruteOptimization scp_opt(api.GetImage("slab_trim"));
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
    }
  else
    {
    // Extract the slab image region
    api.Execute(
          "-clear -push H -cmp -pick 1 -thresh %f %f 1 0 "
          "-push H -times -trim %dvox -as slab "
          "-mf 0x%dx0 -info -slice y 50%% -resample 20%%x20%%x100%% -thresh 0.5 inf 1 0 -swapdim LPI -info -as slab_slice ",
          slab.y0, slab.y1, y_pad, mf_rad);

    // Generate a sparse matrix for faster optimization
    SlabCutPlaneOptimization scp_opt(api.GetImage("slab_slice"));

    // Report
    double t0 = clock();
    printf("Starting optimization set up\n");
    scp_opt.SetupOptimization(0, 2);
    double t1 = clock();
    printf("Optimization set up completed in %f seconds\n", (t1 - t0) / CLOCKS_PER_SEC);

    printf("Starting optimization\n");
    CutPlaneList cpl = scp_opt.RunOptimization();
    double t2 = clock();
    printf("Optimization completed in %f seconds\n", (t2 - t1) / CLOCKS_PER_SEC);
    }

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
          "-thresh 0 inf %d 0 ",
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
        "-scale 0.1 -merge -int 0 -insert F 1 -reslice-identity -push F -replace 1 255 -min "
        "-slice y 50%% -as png_slice -clear");

  // Generate replace commands for R/G/B (like -oli but with in-memory palette)
  for(unsigned int c = 0; c < 3; c++)
    {
    ostringstream oss;
    oss << "-replace ";
    for(unsigned int i = 0; i < 64; i++)
      {
      int val = (palette[i] >> (2-c)) & 0xff;
      oss << (i+1) << " " << val << " ";
      }
    api.Execute("-push png_slice %s", oss.str().c_str());
    }
  api.Execute("-type uchar -foreach -flip y -endfor -omc %s", get_output_filename(param, SLAB_CUT_PNG, slab_id).c_str());
}


int main(int argc, char *argv[])
{
  itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);

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
    else if(arg == "-i" && n_trail > 1)
      {
      param.fn_input_mri = argv[++i];
      param.fn_input_seg = argv[++i];
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

  // Load the data
  typename ReaderType::Pointer r_mask = ReaderType::New();
  r_mask->SetFileName(param.fn_input_seg.c_str());
  r_mask->Update();
  ImagePointer i_mask = r_mask->GetOutput();

  // C3D output should be piped to a log file
  string fn_c3d_log = get_output_filename(param, HEMI_C3D_LOG);
  ofstream c3d_log(fn_c3d_log.c_str());

  // Use c3d to generate a mold
  try 
    { 
    ConvertAPIType api;
    api.RedirectOutput(c3d_log, c3d_log);
    api.AddImage("mask", i_mask);

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
          "-verbose -push mask -swapdim LPI -thresh 1 1 1 0 -as comp_raw "
          "-dilate 1 %dx%dx%d -dilate 0 %dx%dx%d -as comp "
          "-trim 5mm -resample-mm %fmm -as H ",
          param.preproc_dilation, param.preproc_dilation, param.preproc_dilation,
          param.preproc_erosion, param.preproc_erosion, param.preproc_erosion,
          param.mold_resolution_mm);

    // The image H is the trimmed and resampled hemisphere. We will need its dimensions
    // for subsequent work
    ImageType *i_comp_raw = api.GetImage("comp_raw");
    ImageType *i_comp_resampled = api.GetImage("H");
    auto sz = i_comp_resampled->GetBufferedRegion().GetSize();
    auto sp = i_comp_resampled->GetSpacing();
    double off_y = sz[1]*sp[1]/2 - 5;
    double off_x = sz[0]*sp[0]/2 - 5;

    // The next step is printing the main mold, which user may omit
    if(param.flag_print_hemisphere_mold)
      {
      cout << "Generating hemisphere mold" << endl;

      // Figure out the parameters to the mold computation
      double p1 = M_PI * 2.0 / param.slit_spacing_mm;
      double p2 = M_PI * param.slit_width_mm / param.slit_spacing_mm;

      // Create slits along the length of the mold
      api.Execute(
        "-origin-voxel 50%% -cmp -popas z -popas y -popas x "
        "-push y -scale %f -cos -acos -thresh -inf %f -4 4 -as res1", 
        param.mold_resolution_mm, p1, p2);

      api.Execute(
        "-push x -info -thresh %f inf 4 -4 -max "
        "-push y -info -thresh %f %f -4 4 -max "
        "-insert H 1 -copy-transform "
        "-pad 5x5x5 5x5x5 -4 -as mold ",
        off_x, -off_y, off_y);

      // Extrude the brain image and trim extra plastic
      int p3 = (int) ceil(param.mold_wall_thickness_mm / param.mold_resolution_mm);
      api.Execute(
        "-clear -push comp -thresh -inf 0.5 4 -4 "
        "-swapdim RPS -extrude-seg -swapdim LPI -dup "
        "-swapdim LAS -extrude-seg "
        "-thresh -inf 0 0 1 -dilate 0 0x%dx%d -stretch 0 1 4 -4 -swapdim LPI -min "
        "-insert mold 1 -background 4 -reslice-identity "
        "-push mold -min -as carved "
        "-thresh 0 inf 1 0 -o /tmp/b.nii.gz", p3, p3);

      // Get the connected component image and mold
      ImagePointer i_comp = api.GetImage("comp");
      ImagePointer i_mold = api.GetImage("carved");

      save_image(i_comp.GetPointer(), get_output_filename(param, HEMI_VOLUME_IMAGE));
      save_image(i_mold.GetPointer(), get_output_filename(param, HEMI_MOLD_IMAGE));
      }

    // Get the coordinates of the cutting lines used above. In the future
    // we may want to optimize over these instead. 
    //
    // The center line passes through the middle voxel of the image 'H' above
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
    for(int i = 0; i < n_slabs; i++)
      {
      slabs[i].y0 = y_center + (i - n_slabs/2.) * param.slit_spacing_mm;
      slabs[i].y1 = slabs[i].y0 + param.slit_spacing_mm;
      }

    // Process a slab
    for(int i = 0; i < n_slabs; i++)
      {
      if(param.selected_slab < 0 || param.selected_slab == i)
        {
        cout << "Generating mold for slab " << i << " of " << n_slabs << endl;
        process_slab(param, i, slabs[i], i_comp_raw, i_comp_resampled);
        }
      }
    }
  catch(ConvertAPIException &exc)
    {
    cerr << "ConvertAPIException caught: " << exc.what() << endl;
    return -1;
    }

  return 0;
}
