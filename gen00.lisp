(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(setf *features* (union *features* '(:nolog
				     :debug-thread-activity
				     :serial-debug
				     :queue-debug
				     :lock-debug
				     
				     )))
(setf *features* (set-difference *features*
				 '(:nolog
				   :debug-thread-activity
				   :serial-debug
				   :queue-debug
				   
				   )))


(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"../cl-gen2-optix/source/")

  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " "))
  (progn
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)

    (defun set-members (params)
        "setf on multiple member variables of an instance"
        (destructuring-bind (instance &rest args) params
          `(setf ,@(loop for i from 0 below (length args) by 2 appending
                        (let ((keyword (elt args i))
                              (value (elt args (+ i 1))))
                          `((dot ,instance ,keyword) ,value))))))

    (defun set-members-clear (params)
        "setf on multiple member variables of an instance"
        (destructuring-bind (instance &rest args) params
          `(do0
	    (setf ,instance (curly))
	    (setf
	     ,@(loop for i from 0 below (length args) by 2 appending
		    (let ((keyword (elt args i))
			  (value (elt args (+ i 1))))
		      `((dot ,instance ,keyword) ,value)))))))

    (defun ox (code)
      `(progn
	 (let ((res ,code))
	  (declare (type OptixResult res))
	  (unless (== OPTIX_SUCCESS res)
	    ,(logprint (format nil (string "FAIL: optix ~a")
			       (cl-cpp-generator2::emit-c :code code))
		       `(res))))))
    (defun cu (code)
      `(progn
	 (let ((res ,code))
	  (unless (== CUDA_SUCCESS res)
	    ,(logprint (format nil (string "FAIL: cuda ~a")
			       (cl-cpp-generator2::emit-c :code code))
		       `(res))))))
    (defun logprint (msg &optional rest)
      `(do0
	" "
	#-nolog
	(do0
	 ;("std::setprecision" 3)
	 (<< "std::cout"
	     ;;"std::endl"
	     ("std::setw" 10)
	     (dot ("std::chrono::high_resolution_clock::now")
		  (time_since_epoch)
		  (count))
					;,(g `_start_time)
	     
	     (string " ")
	     ("std::this_thread::get_id")
	     (string " ")
	     __FILE__
	     (string ":")
	     __LINE__
	     (string " ")
	     __func__
	     (string " ")
	     (string ,msg)
	     (string " ")
	     ,@(loop for e in rest appending
		    `(("std::setw" 8)
					;("std::width" 8)
		      (string ,(format nil " ~a=" (emit-c :code e)))
		      ,e))
	     "std::endl"
	     "std::flush"))))
    (defun guard (code &key (debug t))
		  `(do0
		    #+lock-debug ,(if debug
		       (logprint (format nil "hold guard on ~a" (cl-cpp-generator2::emit-c :code code))
				 `())
		       "// no debug")
		    #+eou ,(if debug
		     `(if (dot ,code ("std::mutex::try_lock"))
			 (do0
			  (dot ,code (unlock)))
			 (do0
			  ,(logprint (format nil "have to wait on ~a" (cl-cpp-generator2::emit-c :code code))
				     `())))
		     "// no debug")
		    "// no debug"
		   ,(format nil
			    "std::lock_guard<std::mutex> guard(~a);"
			    (cl-cpp-generator2::emit-c :code code))))
    (defun lock (code &key (debug t))
      `(do0
	#+lock-debug ,(if debug
	     (logprint (format nil "hold lock on ~a" (cl-cpp-generator2::emit-c :code code))
		       `())
	     "// no debug")

	#+nil (if (dot ,code ("std::mutex::try_lock"))
	    (do0
	     (dot ,code (unlock)))
	    (do0
	     ,(logprint (format nil "have to wait on ~a" (cl-cpp-generator2::emit-c :code code))
			`())))
	
		    ,(format nil
			     "std::unique_lock<std::mutex> lk(~a);"
			     
				(cl-cpp-generator2::emit-c :code code))
		    ))

    
    (defun emit-globals (&key init)
      (let ((l `((_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
							   (time_since_epoch)
							   (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  `(,name ,type))))))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      (include <chrono>)
	      (defstruct0 State
		  ,@(loop for e in l collect
 			 (destructuring-bind (name type &optional value) e
			   `(,name ,type))))))))
    (defun define-module (args)
      "each module will be written into a c file with module-name. the global-parameters the module will write to will be specified with their type in global-parameters. a file global.h will be written that contains the parameters that were defined in all modules. global parameters that are accessed read-only or have already been specified in another module need not occur in this list (but can). the prototypes of functions that are specified in a module are collected in functions.h. i think i can (ab)use gcc's warnings -Wmissing-declarations to generate this header. i split the code this way to reduce the amount of code that needs to be recompiled during iterative/interactive development. if the module-name contains vulkan, include vulkan headers. if it contains glfw, include glfw headers."
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	  (push `(do0
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  (include "proto2.h")
		  " ")
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header))
	  (push `(:name ,module-name :code (do0 ,@(reverse header) ,module-code))
		*module*))
	(loop for par in global-parameters do
	     (destructuring-bind (parameter-name
				  &key (direction 'in)
				  (type 'int)
				  (default nil)) par
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))
  
  (define-module
      `(main ((_filename :direction 'out :type "char const *")
	      )
	     (do0
	      (include <iostream>
		       <chrono>
		       <cstdio>
		       <cassert>
					;<unordered_map>
		       <string>
		       <fstream>)

	      
	      (let ((state ,(emit-globals :init t)))
		(declare (type "State" state)))


	      (do0
	       (defun mainLoop ()
		 ,(logprint "mainLoop" `())
		 (while (not (glfwWindowShouldClose ,(g `_window)))
		   (glfwPollEvents)
		   (drawFrame)
		   (drawGui)
		   (glfwSwapBuffers ,(g `_window))
		   )
		 ,(logprint "exit mainLoop" `()))
	       (defun run ()
		 ,(logprint "start run" `())
		 
		 (initWindow)
		 (initGui)
		 
		 (initDraw)
		 (initOptix)
		 
		 (mainLoop)
		 ,(logprint "finish run" `())))
	      
	      (defun main ()
		(declare (values int))
		
		(setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
					     (time_since_epoch)
					     (count)))
		,(logprint "start main" `())
		(setf ,(g `_filename)
		      (string "bla.txt"))

		(do0
		 (run)
		 ,(logprint "start cleanups" `())
		 (cleanupOptix)
		 (cleanupDraw)
		 (cleanupGui)
		 (cleanupWindow)
		)
		,(logprint "end main" `())
		(return 0)))))

  
  
  
  (define-module
      `(glfw_window
	((_window :direction 'out :type GLFWwindow* )
	 (_framebufferResized :direction 'out :type bool)
	 )
	(do0
	 (defun keyCallback (window key scancode action mods)
	   (declare (type GLFWwindow* window)
		    (type int key scancode action mods))
	   (when (and (or (== key GLFW_KEY_ESCAPE)
			  (== key GLFW_KEY_Q))
		      (== action GLFW_PRESS))
	     (glfwSetWindowShouldClose window GLFW_TRUE)))
	 (defun errorCallback (err description)
	   (declare (type int err)
		    (type "const char*" description))
	   ,(logprint "error" `(err description)))
	 (defun framebufferResizeCallback (window width height)
	   (declare (values "static void")
		    ;; static because glfw doesnt know how to call a member function with a this pointer
		    (type GLFWwindow* window)
		    (type int width height))
	   ,(logprint "resize" `(width height))
	   (let ((app ("(State*)" (glfwGetWindowUserPointer window))))
	     (setf app->_framebufferResized true)))
	 (defun initWindow ()
	   (declare (values void))
	   (when (glfwInit)
	     (do0
	      (glfwSetErrorCallback errorCallback)
	      
	      (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 2)
	      (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0)
	      
	      (glfwWindowHint GLFW_RESIZABLE GLFW_TRUE)
	      (setf ,(g `_window) (glfwCreateWindow 930 930
						    (string "vis window")
						    NULL
						    NULL))
	      ,(logprint "initWindow" `(,(g `_window)
					 (glfwGetVersionString)))
	      ;; store this pointer to the instance for use in the callback
	      (glfwSetKeyCallback ,(g `_window) keyCallback)
	      (glfwSetWindowUserPointer ,(g `_window) (ref state))
	      (glfwSetFramebufferSizeCallback ,(g `_window)
					      framebufferResizeCallback)
	      (glfwMakeContextCurrent ,(g `_window))
	      (glfwSwapInterval 1)
	      )))
	 (defun cleanupWindow ()
	   (declare (values void))
	   (glfwDestroyWindow ,(g `_window))
	   (glfwTerminate)))))
  (define-module
      `(draw ((_fontTex :direction 'out :type GLuint))
	     (do0
	      (include <algorithm>)
	      (defun uploadTex (image w h)
		(declare (type "const void*" image)
			 (type int w h))
		(glGenTextures 1 (ref ,(g `_fontTex)))
		(glBindTexture GL_TEXTURE_2D ,(g `_fontTex))
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR)
		(glTexImage2D GL_TEXTURE_2D 0 GL_RGBA w h 0 GL_RGBA GL_UNSIGNED_BYTE image))
	      
	      (defun initDraw ()
					;(glEnable GL_TEXTURE_2D)
		#+nil (glEnable GL_DEPTH_TEST)
		#+nil (glHint GL_LINE_SMOOTH GL_NICEST)
		#+nil (do0 (glEnable GL_BLEND)
		     (glBlendFunc GL_SRC_ALPHA
				  GL_ONE_MINUS_SRC_ALPHA))
		(glClearColor 0 0 0 1))
	      (defun cleanupDraw ()
		(glDeleteTextures 1 (ref ,(g `_fontTex))))
	      (defun drawFrame ()
		(glClear (logior GL_COLOR_BUFFER_BIT
				 GL_DEPTH_BUFFER_BIT))))))
  (define-module
      `(gui ()
	    (do0
	     "// https://youtu.be/nVaQuNXueFw?t=317"
	     "// https://blog.conan.io/2019/06/26/An-introduction-to-the-Dear-ImGui-library.html"
	     (include "imgui/imgui.h"
		      "imgui/imgui_impl_glfw.h"
		      "imgui/imgui_impl_opengl2.h")
	     (include <algorithm>
		      <string>)
	     (defun initGui ()
	       ,(logprint "initGui" '())
	       (IMGUI_CHECKVERSION)
	       ("ImGui::CreateContext")
	       
	       (ImGui_ImplGlfw_InitForOpenGL ,(g `_window)
					     true)
	       (ImGui_ImplOpenGL2_Init)
	       ("ImGui::StyleColorsDark"))
	     (defun cleanupGui ()
	       (ImGui_ImplOpenGL2_Shutdown)
	       (ImGui_ImplGlfw_Shutdown)
	       ("ImGui::DestroyContext"))
	     
	     (defun drawGui ()
	       #+nil (<< "std::cout"
		   (string "g")
		   "std::flush")
	       
	       (ImGui_ImplOpenGL2_NewFrame)
	       (ImGui_ImplGlfw_NewFrame)
	       ("ImGui::NewFrame")
	       	       
	       (let ((b true))
		      ("ImGui::ShowDemoWindow" &b))
	       ("ImGui::Render")
	       (ImGui_ImplOpenGL2_RenderDrawData
		("ImGui::GetDrawData"))
	       ))))

  
  (define-module
      `(optix (
	       ;; createContext
	       (dev_id :type "const int")
	       (stream :type CUstream)
	       (dev_prop :type cudaDeviceProp)
	       (cuctx :type CUcontext)
	       (oxctx :type OptixDeviceContext)
	       ;; createModule
	       (module_compile_options :type OptixModuleCompileOptions)
	       (pipeline_compile_options :type OptixPipelineCompileOptions)
	       (pipeline_link_options :type OptixPipelineLinkOptions)

	       ;;
	       (pipeline :type OptixPipeline)
	       (module :type OptixModule)
	       ;;

	       ,@(loop for e in `(raygen miss hitgroup)
		    appending
		      `((,(format nil "~a_programs" e) :type "std::vector<OptixProgramGroup>")
			(,(format nil "~a_records_buffer" e) :type CUDABuffer))
		      )
	       
	       ;(hitgroup_records_buffer :type CUDABuffer)
	       (shader_bindings_table :type OptixShaderBindingTable)
	       ;;
	       (launch_params :type LaunchParams)
	       (launch_params_buffer :type CUDABuffer)
	       ;;
	       (color_buffer :type CUDABuffer)
	       )
	      (do0
	       "// derived from Ingo Wald's optix7course example03_inGLFWindow SampleRenderer.cpp"
	       (include <cuda_runtime.h>
			<optix.h>
			<optix_stubs.h>
			<optix_function_table_definition.h>)
	       " "
					
	       "extern \"C\" const  char ptx_code[];"

	     (defun createContext ()
	       (declare (type "static void"))
	       ,(cu `(cudaSetDevice ,(g `dev_id)))
	       ,(cu `(cudaStreamCreate (ref ,(g `stream))))
	       (cudaGetDeviceProperties (ref ,(g `dev_prop))
					,(g `dev_id))
	       ,(logprint "running on device:"
			  `(,(g `dev_prop.name)))
	       ,(cu `(cuCtxGetCurrent (ref ,(g `cuctx))))
	       ,(ox
		 `(optixDeviceContextCreate
		   ,(g `cuctx) 0 (ref ,(g `oxctx))))
	       (let ((log_cb (lambda (level
			    tag
			    msg
			    data)
		     (declare
		      (type "unsigned int" level)
		      (type "const char*"
			    tag
			    msg)
		      (type "void*"
			    data))
		     ,(logprint "context_log"
				`(level
				  tag
				  msg)))))
		,(ox
		  `(optixDeviceContextSetLogCallback
		    ,(g `oxctx)
		   log_cb
		   nullptr 4))))
	     (defun createModule ()
	       ,(set-members-clear`(,(g `module_compile_options)
				    :maxRegisterCount 50
				    :optLevel OPTIX_COMPILE_OPTIMIZATION_DEFAULT
				     :debugLevel OPTIX_COMPILE_DEBUG_LEVEL_NONE))
	       ,(set-members-clear`(,(g `pipeline_compile_options)
				     :traversableGraphFlags OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
				     :usesMotionBlur false
				     :numPayloadValues 2
				     :numAttributeValues  2
				     :exceptionFlags     OPTIX_EXCEPTION_FLAG_NONE
				     :pipelineLaunchParamsVariableName (string "optixLaunchParams") 
      
				     ))
	       ,(set-members-clear`(,(g `pipeline_link_options)
				           
				     :overrideUsesMotionBlur  false
				     :maxTraceDepth           2))

	       
	       (let ((ptx ptx_code)
		     (log[2048])
		     (size_log (sizeof log)))
		 (declare (type char log[2048])
			  (type "const std::string" ptx))
		 ,(ox `(optixModuleCreateFromPTX
			,(g `oxctx)
			(ref ,(g `module_compile_options))
			(ref ,(g `pipeline_compile_options))
					;ptx_code
			(ptx.c_str)
			;(strlen ptx_code)
			(ptx.size)
			log
			&size_log
			(ref ,(g `module))))
		 (when (< 1 size_log)
		   ,(logprint "" `(size_log log)))))
	     ,@(loop for line in
		    `((raygen RayGen "__raygen__renderFrame")
		      (miss Miss "__miss__radiance")
		      (hitgroup HitGroup)) collect
		    (destructuring-bind  (e f &optional entry) line
		      (let ((var (g (format nil "~a_programs" e))))
		       `(defun ,(format nil "create~aPrograms" f) ()
			  (dot ,var (resize 1))
			  (let ((pg_options)
				(pg_desc))
			    (declare (type OptixProgramGroupOptions pg_options)
				     (type OptixProgramGroupDesc pg_desc))
			    ,(set-members-clear `(pg_options))
			    ,(set-members-clear `(pg_desc
						  :kind ,(string-upcase (format nil "OPTIX_PROGRAM_GROUP_KIND_~a" f))
						  
						  ,@(if (member e `(raygen miss))
							`(,(intern (string-upcase (format nil "~a.module" (string-downcase f)))) ,(g `module)
							   ,(intern (format nil "~a.entryFunctionName"
									   (string-downcase f)))
							   (string ,entry))
							`(,(intern (format nil "~a.moduleCH" (string-downcase f))) ,(g `module)
							   ,(intern (format nil "~a.moduleAH" (string-downcase f))) ,(g `module)
							   ,(intern (format nil "~a.entryFunctionNameCH"
									   (string-downcase f)))
							   (string "__closesthit__radiance")
							   ,(intern (format nil "~a.entryFunctionNameAH"
									    (string-downcase f)))
							   (string "__anyhit__radiance")))))
			    (let ((log[2048])
				  (size_log (sizeof log)))
			      (declare (type char log[2048])
				       )
			      ,(ox `(optixProgramGroupCreate
				     ,(g `oxctx)
				     (ref pg_desc)
				     1
				     (ref pg_options)
				     log
				     &size_log
				     (ref (aref ,var 0))))
			      (when (< 1 size_log)
				,(logprint "" `(size_log log)))))))))
	     (defun createPipeline ()
	       (let ((program_groups))
		 (declare (type "std::vector<OptixProgramGroup>" program_groups))
		 ,@(loop for e in `(raygen miss hitgroup) collect
			`(foreach (p ,(g (format nil "~a_programs" e)))
				  (program_groups.push_back p)))
		 (let ((log[2048])
		       (size_log (sizeof log)))
		   (declare (type char log[2048]))
			      ,(ox `(optixPipelineCreate
				     ,(g `oxctx)
				     (ref ,(g `pipeline_compile_options))
				     (ref ,(g `pipeline_link_options))
				     (program_groups.data)
				     (static_cast<int> (program_groups.size))
				     log
				     &size_log
				     (ref ,(g `pipeline))))
			      (when (< 1 size_log)
				,(logprint "" `(size_log log))))
		 ,(ox `(optixPipelineSetStackSize
			,(g `pipeline)
			(* 2 1024) ;; direct, invoked from ISor AH
 			(* 2 1024) ;; direct, invoked from RG MSor CH
			(* 2 1024) ;; continuation
			1 ;; maximum depth of traversable graph passed to trace
			))))
	     ,@(loop for e in `(raygen miss hitgroup) collect
		    `(defstruct0
			 ,(format nil "__align__(OPTIX_SBT_RECORD_ALIGNMENT) ~a_record_t" e)
			 ((aref header OPTIX_SBT_RECORD_HEADER_SIZE)
			  "__align__(OPTIX_SBT_RECORD_ALIGNMENT) char")
		       (data "void*")
		 ))
	     (defun buildSBT ()
	       ,@(loop for e in `(raygen miss hitgroup) collect
		      (let ((records (format nil "~a_records" e))
			    (progs (g (format nil "~a_programs" e)))
			    (buffer (g (format nil "~a_records_buffer" e)))
			    (type (format nil "~a_record_t" e)))
			`(let ((,records))
			   (declare (type ,(format nil "std::vector<~a>" type)  ,records))
			 (dotimes (i (dot ,progs (size)))
			   (let ((rec))
			     (declare (type ,type rec))
			     ,(ox `(optixSbtRecordPackHeader (aref ,progs i)
							     (ref rec)))
			     (dot ,records (push_back rec))))
			 (dot ,buffer
			      (alloc_and_upload ,records))
			 ,(cond
			    ((eq e 'raygen)
			     `(setf (dot ,(g `shader_bindings_table)
					,(format nil "~aRecord" e))
				   (dot ,buffer
					(d_pointer))))
			    (t
			     `(do0
			       (setf (dot ,(g `shader_bindings_table)
					  ,(format nil "~aRecordBase" e))
				     (dot ,buffer
					  (d_pointer)))
			       (setf (dot ,(g `shader_bindings_table)
					  ,(format nil "~aRecordStrideInBytes" e))
				     (sizeof ,type))
			       (setf (dot ,(g `shader_bindings_table)
					  ,(format nil "~aRecordCount" e))
				     (static_cast<int> (dot ,records
							    (size))))))))))
	       )
	     
	     (defun initOptix ()
	       ,(logprint "initOptix" '())
	       (cudaFree 0)
	       (let ((num_devices))
		 (declare (type int num_devices))
		 (cudaGetDeviceCount &num_devices))
	       (when (== 0 num_devices)
		 ,(logprint (string "FAIL: no cuda device")))
	       ,(ox `(optixInit))
	       (createContext)
	       (createModule)
	       (createRayGenPrograms)
	       (createMissPrograms)
	       (createHitGroupPrograms)
	       (createPipeline)
	       (buildSBT)
	       )
	     (defun cleanupOptix ()
	       )
	     )))

  (define-module
      `(cuda_device_programs
	()
	(do0
	 (include <optix_device.h>)
	 " "

	 #+nil (include "LaunchParams.h")
	 #+nil
	 (defstruct0 LaunchParams
			(frameID int)
		      (colorBuffer uint32_t*)
		      (fbSize_x int)
		      (fbSize_y int)
		      ;(fbSize vec2i)
		      )

	 
	 (let ((optixLaunchParams))
	   (declare (type "extern \"C\" __constant__ LaunchParams" optixLaunchParams)))
	 (defun __closesthit__radiance ()
	   (declare (values "extern \"C\" __global__ void")))
	 (defun __anyhit__radiance ()
	   (declare (values "extern \"C\" __global__ void")))
	 (defun __miss__radiance ()
	   (declare (values "extern \"C\" __global__ void")))
	 (defun __raygen__renderFrame ()
	   (declare (values "extern \"C\" __global__ void"))
	   (let ((frameID optixLaunchParams.frameID)
		 (ix (dot (optixGetLaunchIndex) x))
		 (iy (dot (optixGetLaunchIndex) y))
		 (fbIndex (+ ix
			     (* iy optixLaunchParams.fbSize_x))))
	     (declare (type "const int" frameID)))
	   (setf (aref optixLaunchParams.colorBuffer fbIndex)
		 (hex #xff123456)))
	 
	 )))

  
  (progn
    (with-open-file (s (asdf:system-relative-pathname 'cl-cpp-generator2
						      (merge-pathnames #P"proto2.h"
								       *source-dir*))
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       (unless cuda
		 (emit-c :code code :hook-defun 
			 #'(lambda (str)
			     (format s "~a~%" str))))
	       
	       (write-source (asdf:system-relative-pathname
			      'cl-cpp-generator2
			      (format nil
				      "~a/vis_~2,'0d_~a.~a"
				      *source-dir* i name
				      (if cuda
					  "cu"
					  "cpp")))
			     code)))))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"utils.h"
				    *source-dir*))
		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    (include <vector>
			     <array>
			     <iostream>
			     <iomanip>)
		    
		    " "
		    (do0
		     
		     " "
		     ,@(loop for e in (reverse *utils-code*) collect
			  e)
			
		     
		     " "
		     
		     )
		    " "
		    "#endif"
		    " "))
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"globals.h"
								     *source-dir*))
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "
		    (include <GLFW/glfw3.h>)
		    
		    " "
		    " "
		    (include <thread>
			     <mutex>
			     <queue>
			     <deque>
			     <string>
			     <condition_variable>
			     <complex>)


		    (include <cuda_runtime.h>
			     <optix.h>
			     <optix_stubs.h>)

		    (include <cassert>)

		    (defstruct0 LaunchParams
			(frameID int)
		      (colorBuffer uint32_t*)
		      (fbSize_x int)
		      (fbSize_y int)
		      ;(fbSize vec2i)
		      )

		    (do0
		     (defclass CUDABuffer ()
		       "public:"
		       (let ((_d_ptr)
			     (_size_in_bytes))
			 (declare (type void* _d_ptr)
				  (type size_t _size_in_bytes)))
		       (defun d_pointer ()
			 (declare (values CUdeviceptr))
			 (return (reinterpret_cast<CUdeviceptr> _d_ptr)))
		       (defun resize (size)
			 (declare (type size_t size))
			 (when _d_ptr
			   (free))
			 (alloc size))
		       (defun alloc (size)
			 (declare (type size_t size))
			 (assert (== nullptr _d_ptr))
			 (setf this->_size_in_bytes size)
			 ,(cu `(cudaMalloc (static_cast<void**> &_d_ptr)
					  _size_in_bytes)))
		       (defun free ()
			 ,(cu `(cudaFree _d_ptr))
			 (setf _d_ptr nullptr
			       _size_in_bytes 0))
		       (defun alloc_and_upload (vt)
			 (declare (type "const std::vector<T>&" vt)
				  (values "template<typename T> void"))
			 (alloc (* (vt.size)
				   (sizeof T)))
			 (upload ("static_cast<const T*>" (vt.data))
				 (vt.size)))
		       (defun upload (dat count)
			 (declare (type "const T*" dat)
				  (type size_t count)
				  (values "template<typename T> void"))
			 (assert (!= nullptr _d_ptr))
			 (assert (== _size_in_bytes (* count (sizeof T))))
			 ,(cu `(cudaMemcpy _d_ptr
					   ("static_cast<const void*>" dat)
					  (* count (sizeof T))
					  cudaMemcpyHostToDevice)))
		       (defun download (dat count)
			 (declare (type "T*" dat)
				  (type size_t count)
				  (values "template<typename T> void"))
			 (assert (!= nullptr _d_ptr))
			 (assert (== _size_in_bytes (* count (sizeof T))))
			 ,(cu `(cudaMemcpy  (static_cast<void*> dat)
					  _d_ptr
					  (* count (sizeof T))
					  cudaMemcpyDeviceToHost)))))
		    
		    (do0
		     "template <typename T, int MaxLen>"
		     (defclass FixedDequeTM "public std::deque<T>"
		       "// https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue"
		       
		       "public:"
		       (let ((mutex))
			 (declare (type "std::mutex" mutex)))
		       (defun push_back (val)
			 (declare (type "const T&" val))
			 (when (== MaxLen (this->size))
			   (this->pop_front))
			 ("std::deque<T>::push_back" val))))
		    
		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))))

