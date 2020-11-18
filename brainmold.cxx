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

using namespace std;
// using namespace gnlp;

// Define common types
typedef ConvertAPI<double, 3> ConvertAPIType;
typedef typename ConvertAPIType::ImageType ImageType;
typedef typename ImageType::Pointer ImagePointer;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;

class Parameters {
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

};


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
    );

  return -1;
}

template <typename TImage>
void save_image(TImage *img, const Parameters &param, const char *suffix)
{
  typedef itk::ImageFileWriter<TImage> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(img);

  char fn_out[1024];
  sprintf(fn_out, "%s/%s_%s.nii.gz", 
    param.fn_output_root.c_str(), param.id_string.c_str(), suffix);

  writer->SetFileName(fn_out);
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


void process_slab(Parameters &param, Slab &slab, ImageType *i_hemi_raw, ImageType *i_hemi_mask)
{
  // Create our own API
  ConvertAPIType api;

  // Deal with the raw image (for visualization)
  api.AddImage("raw", i_hemi_raw);
  api.Execute("-verbose -push raw -cmp -pick 1 -thresh %f %f 1 0 "
              "-push raw -times -trim 2vox -as slab_raw -o /tmp/slab_raw.nii.gz",
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

  // Generate a mold for this slab

  // Compute coordinate maps and clear the stack
  api.Execute("-push slab -cmp -popas z -popas y -popas x -clear");

  // Apply each cutplane
  for(auto &cp : cpl)
    {
    cout << "Cut plane: r = " << cp.first << "   theta = " << cp.second << endl;
    api.Execute(
          "-push x -scale %f -push z -scale %f -add -shift %f -abs "
          "-stretch 0 %f -4 4 -clip -4 4 ",
          cos(cp.second), sin(cp.second), -cp.first, 2 * param.slit_width_mm);
    }

  // Combine the cut planes into one image
  api.Execute("-accum -min -endaccum -o /tmp/my_cuts.nii.gz -as cuts");

  // Figure out where the base starts: 5vox from the bottom or top.
  api.Execute("-push y -thresh %f inf 4 -4 -max -as cutsbase -o /tmp/my_cuts_base.nii.gz", slab.y1);

  // Apply extrusion to the slab itself
  int dilation = (int) ceil(param.mold_wall_thickness_mm / param.mold_resolution_mm);
  api.Execute(
        "-push slab -thresh -inf 0.5 4 -4 -swapdim ARS -extrude-seg -swapdim LPI "
        "-dup -swapdim PRS -extrude-seg -thresh -inf 0 0 1 -dilate 0 0x%dx%d "
        "-stretch 0 1 4 -4 -swapdim LPI -min "
        "-push cutsbase -min -as carved "
        "-thresh 0.0 inf 1 0 -pad 2x2x2 2x2x2 0 -o /tmp/slab_carved.nii.gz", dilation, dilation);

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

  // Use c3d to generate a mold
  try 
    { 
    ConvertAPIType api;
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
    api.Execute(
          "-verbose -push mask -swapdim LPI -thresh 1 1 1 0 -as comp_raw -o /tmp/comp_raw.nii.gz "
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

      save_image(i_comp.GetPointer(), param, "seg_conn_comp");
      save_image(i_mold.GetPointer(), param, "seg_mold");
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
    unsigned int n_slabs = (unsigned int) ceil(off_y / param.slit_spacing_mm) * 2;
    printf("Slabs %d, center slab coordinates: %f\n", n_slabs, y_center);

    // Create the slabs
    std::vector<Slab> slabs(n_slabs);
    for(unsigned int i = 0; i < n_slabs; i++)
      {
      slabs[i].y0 = y_center + (i - n_slabs/2.) * param.slit_spacing_mm;
      slabs[i].y1 = slabs[i].y0 + param.slit_spacing_mm;
      }

    // Process a slab
    printf("%f, %f\n", slabs[8].y0, slabs[8].y1);
    process_slab(param, slabs[8], i_comp_raw, i_comp_resampled);
   
    }
  catch(ConvertAPIException &exc)
    {
    cerr << "ConvertAPIException caught: " << exc.what() << endl;
    return -1;
    }

  return 0;
}
