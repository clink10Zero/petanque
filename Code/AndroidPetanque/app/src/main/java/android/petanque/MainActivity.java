package android.petanque;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.icu.text.SimpleDateFormat;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.FileProvider;

import static android.graphics.Bitmap.Config.ARGB_8888;
import static android.graphics.Bitmap.Config.RGBA_F16;
import static android.graphics.Bitmap.Config.RGB_565;
import static android.graphics.Color.BLUE;
import static android.graphics.Color.GREEN;
import static android.graphics.Color.RED;
import static java.lang.Math.floor;
import static java.lang.Math.pow;
import static java.lang.Math.round;
import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;
import static org.opencv.core.Core.absdiff;
import static org.opencv.core.Core.merge;
import static org.opencv.core.Core.normalize;
import static org.opencv.core.Core.split;
import static org.opencv.core.CvType.CV_16UC3;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.Canny;
import static org.opencv.imgproc.Imgproc.GaussianBlur;
import static org.opencv.imgproc.Imgproc.HOUGH_GRADIENT;
import static org.opencv.imgproc.Imgproc.HoughCircles;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.bilateralFilter;
import static org.opencv.imgproc.Imgproc.circle;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.dilate;
import static org.opencv.imgproc.Imgproc.line;
import static org.opencv.imgproc.Imgproc.medianBlur;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.resize;


public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAPTURE_IMAGE = 1;
    private static final int PICK_IMAGE = 2;
    String imagePath;

    private static String TAG = "MainActivity";
    static {
        if(OpenCVLoader.initDebug()){
            Log.d(TAG, "success");
        }
        else{
            Log.d(TAG,"not installed");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button takePhotoButton = findViewById(R.id.btn_prendre_photo_renseingner_annexe);
        Button browseButton = findViewById(R.id.btn_parcourir_rensgeiner_annexe);
        Button btnTraiter = findViewById(R.id.btn_traiter);

        Activity activity = this;
        takePhotoButton.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                int readPermission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE);
                int writePermission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
                if (writePermission != PackageManager.PERMISSION_GRANTED || readPermission != PackageManager.PERMISSION_GRANTED) {
                    activity.requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_CAPTURE_IMAGE);
                }
                else {
                    openCameraIntent();
                }
            }
        });
        browseButton.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                int readPermission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE);
                int writePermission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

                if (writePermission != PackageManager.PERMISSION_GRANTED || readPermission != PackageManager.PERMISSION_GRANTED) {
                    activity.requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE}, PICK_IMAGE);
                }
                else{
                    Intent intent = new Intent();
                    intent.setType("image/*");
                    intent.setAction(Intent.ACTION_GET_CONTENT);
                    startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE);
                }
            }
        });

        btnTraiter.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                Mat imageOr = imread(imagePath);
                int scale_percent = 20;
                int width = (imageOr.cols() * scale_percent / 100);
                int height = (imageOr.rows() * scale_percent / 100);
                Size s = new Size(width,height);
                Mat image = new Mat();
                resize(imageOr,image,s,0.2,0.2,INTER_AREA);

                Mat graytmp = new Mat(width,height,CV_16UC3);
                Mat gray = new Mat(width,height,CV_16UC3);
                cvtColor(image,gray,COLOR_BGR2GRAY);
                medianBlur(gray,graytmp, 5);
                bilateralFilter(graytmp,gray, 4, 30, 60);


                Mat edges = new Mat(width,height,CV_16UC3);
                Canny(gray,edges, 60, 120);
                Mat circles = new Mat(width,height,CV_16UC3);
                HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 30, 140, 20, 0, 40);


                Mat circlesint = new Mat(width,height,CV_16UC3);
                circles.convertTo(circlesint,0);
                double[] cochonnet = circles.get(0,0);
                int tailleCochonnet = 3;

                //recherche du cochonnet
                for(int k = 0; k<circles.cols(); k++){
                    double[] i = circles.get(0,k);
                    if( i[2] < cochonnet[2]){
                        cochonnet = i;
                    }
                }
                 double taillepixel = tailleCochonnet/(cochonnet[2] * 2);

                //affichage et calcule de distance
               for(int k = 0; k<circles.cols(); k++){
                    double[] i = circles.get(0,k);
                    if (i[0] != cochonnet[0] & i[1] != cochonnet[1]) {
                        circle(image, new Point(i[0], i[1]), (int)round(i[2]), new Scalar(0,255,0),2);
                        circle(image, new Point(i[0], i[1]), 2, new Scalar(255,0,0),3);

                        double x = pow(i[0] - cochonnet[0], 2) * pow(taillepixel, 2);
                        double y = pow(i[1] - cochonnet[1], 2) * pow(taillepixel, 2);
                        double distance = Math.sqrt(x + y);

                       line(image,new Point(cochonnet[0], cochonnet[1]), new Point(i[0], i[1]), new Scalar(0,0,255),3);
                       putText(image, (""+distance ),new Point (i[0], i[1]), FONT_HERSHEY_SIMPLEX, 2, new Scalar(0, 0, 0),1, 2);
                    }
                }

                Bitmap bmp = Bitmap.createBitmap(image.cols(),image.rows(),RGB_565);
                Utils.matToBitmap(image, bmp,false);
                displayPhoto(activity,bmp);
            }
        });
    }

    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        activityResult(this,resultCode,requestCode,data);
        Button btnTraiter = findViewById(R.id.btn_traiter);
        if(imagePath!=null){
            btnTraiter.setVisibility(View.VISIBLE);
        }
    }

    private File createImageFile() throws IOException {
        String timeStamp =
                new SimpleDateFormat("yyyyMMdd_HHmmss",
                        Locale.getDefault()).format(new Date());
        String imageFileName = "IMG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);

        imagePath = image.getAbsolutePath();
        return image;
    }

    private void openCameraIntent() {
        Intent pictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (pictureIntent.resolveActivity(getPackageManager()) != null) {
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
            if (photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(this, "android.petanque.provider", photoFile);
                pictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(pictureIntent,
                        REQUEST_CAPTURE_IMAGE);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            switch (requestCode) {
                case REQUEST_CAPTURE_IMAGE:
                    openCameraIntent();
                    break;
                case PICK_IMAGE:
                    Intent intent = new Intent();
                    intent.setType("image/*");
                    intent.setAction(Intent.ACTION_GET_CONTENT);
                    startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE);
                    break;
            }
        }
        else {
            startActivity(new Intent(this,MainActivity.class));
        }
    }

    public void activityResult(final Activity activity, int resultCode, int requestCode, @Nullable Intent data) {
        if (resultCode != RESULT_CANCELED) {
            if (requestCode == PICK_IMAGE) {//retour de galerie
                Uri selectedImage = null;
                if (data != null) {
                    selectedImage = data.getData();
                }
                String[] filePathColumn = {MediaStore.Images.Media.DATA};

                Cursor cursor = null;
                if (selectedImage != null) {
                    String sel = MediaStore.Images.Media._ID + "=?";

                    String wholeID = DocumentsContract.getDocumentId(selectedImage);

                    String id = wholeID.split(":")[1];
                    cursor = activity.getContentResolver().
                            query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                    filePathColumn, sel, new String[]{id}, null);
                }
                if (cursor != null) {
                    if (cursor.moveToFirst()) {

                        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                        imagePath = cursor.getString(columnIndex);
                        cursor.close();

                        displayPhoto(activity,imagePath);
                    }
                }
            }
            if (requestCode == REQUEST_CAPTURE_IMAGE) {//retour d'appareil photo
                File[] filePhotoList = Objects.requireNonNull(activity.getExternalFilesDir(Environment.DIRECTORY_PICTURES)).listFiles();//photos take by the app
                File src = filePhotoList[filePhotoList.length - 1];

                String directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)+"/PetanquePictures";
                String timeStamp =
                        new SimpleDateFormat("yyyyMMdd_HHmmss",
                                Locale.getDefault()).format(new Date());
                String imageFileName = "IMG_" + timeStamp + ".jpg";
                File fileDirectory = new File(directory);
                if (!fileDirectory.exists()) {
                    fileDirectory.mkdirs();
                }
                File dest = new File(directory + "/" + imageFileName);

                try {
                    copy(src, dest);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                src.delete();

                displayPhoto(activity,directory + "/" + imageFileName);
                imagePath = directory + "/" +imageFileName;
            }
        }
    }


    private static void copy(File src, File dst) throws IOException {
        try (InputStream in = new FileInputStream(src)) {
            try (OutputStream out = new FileOutputStream(dst)) {
                byte[] buf = new byte[1024];
                int len;
                while ((len = in.read(buf)) > 0) {
                    out.write(buf, 0, len);
                }
            }
        }
    }

    public static void displayPhoto(final Activity activity, String photoPath) {
        Bitmap bitmap = BitmapFactory.decodeFile(photoPath);
        if (bitmap == null) {
            Toast.makeText(activity, "Photo inexistante", Toast.LENGTH_SHORT).show();
        } else {
            final String picturePath = photoPath;
            Bitmap photoBitmap = BitmapFactory.decodeFile(picturePath);

            ImageView view = activity.findViewById(R.id.photo);
            view.setImageBitmap(Bitmap.createScaledBitmap(photoBitmap,(int)(photoBitmap.getWidth()*0.2),(int)(photoBitmap.getHeight()*0.2),true));
        }
    }

    public static void displayPhoto(final Activity activity, Bitmap bitmap) {
        ImageView view = activity.findViewById(R.id.photo);
        view.setImageBitmap(bitmap);
    }
}

