import { Link } from "react-router";
import { SiGithub, SiHuggingface } from "react-icons/si";
import { FileText, Database } from "lucide-react";
import { Button } from "./ui/button";

export function Footer() {
  return (
    <footer className="w-full border-t bg-footer-background/70 py-6 md:py-10 flex flex-col items-center">
      <div className="container mx-auto flex flex-col items-center gap-6 px-6 xl:max-w-4xl">
        <div className="flex flex-wrap justify-center gap-3">
          <Button variant="ghost" size="sm" asChild>
            <a
              href="https://openaccess.thecvf.com/content/CVPR2026F/papers/Ide_Beyond_Single_Object_Learning_3D_Relations_with_Large_Language_Models_CVPRF_2026_paper.pdf"
              target="_blank"
              rel="noopener noreferrer"
            >
              <FileText className="mr-2 h-4 w-4" />
              Paper
            </a>
          </Button>
          <Button variant="ghost" size="sm" asChild>
            <a
              href="https://github.com/KohsukeIde/BeyondSingleObject"
              target="_blank"
              rel="noopener noreferrer"
            >
              <SiGithub className="mr-2 h-4 w-4" />
              Code
            </a>
          </Button>
          <Button variant="ghost" size="sm" asChild>
            <a
              href="https://huggingface.co/idekoh/Multi-3DLLM"
              target="_blank"
              rel="noopener noreferrer"
            >
              <SiHuggingface className="mr-2 h-4 w-4" />
              Models
            </a>
          </Button>
          <Button variant="ghost" size="sm" asChild>
            <a
              href="https://huggingface.co/datasets/idekoh/BeyondSingleObject"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Database className="mr-2 h-4 w-4" />
              Dataset
            </a>
          </Button>
        </div>
        <div className="flex flex-col items-center gap-2 text-center">
          <Link to="/" className="flex items-center space-x-2">
            <span className="font-bold text-lg">Beyond Single Object</span>
          </Link>
          <p className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} Beyond Single Object Project. All rights
            reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
