'use strict';

import {
    Popup,
    Div, 
    Api,
    Form,
    FormField,
    TextInput,
    FileInput
} from '@src/classes';

export class AddDocumentPopup extends Popup {

    private form: Form;

    private title: FormField;
    private description: FormField;
    private file: FormField;

    constructor(private teamId: number) {

        super({
            validText: 'Upload',
            cancellable: true,
            title: `Upload document`,
        });

        this.build();
    }

    /*
    **
    **
    */
    private async build() : Promise<void> {

        this.form = new Form(this.content);

        //=====
        //TITLE
        //=====

        this.title = this.form.add(new TextInput({
            label: 'Title',
            mandatory: true
        }));

        this.title.linkToErrorKey('title');

        //===========
        //DESCRIPTION
        //===========

        this.description = this.form.add(new TextInput({
            label: 'Description',
            mandatory: true
        }));

        this.description.linkToErrorKey('description');

        //====
        //FILE
        //====

        this.file = this.form.add(new FileInput({
            label: 'File'
        }));

        this.file.linkToErrorKey('file');

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/team/documents/add', {
                teamId: this.teamId,
                title: this.title.input.getValue(),
                description: this.description.input.getValue(),
                fileName: this.file.input.getName(),
                file: this.file.input.getBase64()
            });

            this.emit('success');
            this.hide();
                        
        } catch(error: any) {

            this.form.displayError(error);
            this.validButton.unload();
        }
    }
}