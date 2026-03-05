'use strict';

import {
    Div,
    Popup,
    Form,
    FormField,
    TextInput,
    PasswordInput,
    Api,
    ClientLocation,
    Tools,
    PBKDF2,
    ImageDiv,
} from '@src/classes';

export class LoginPopup extends Popup {

    static readonly HASH_DERIVATION_KEY_BYTE_LENGTH = 32;
    static readonly HASH_ALGORITHM = 'SHA-256';

    private form: Form;

    private email: FormField;
    private password: FormField;

    constructor() {

        super({
            validText: 'Sign in',
            title: `Sign in`,
            closeZoneHidden: true,
            notRemovable: true
        });

        this.addClass('login small');
        
        this.drawAppData();
        this.drawForm();
    }

    /*
    **
    **
    */
    private async drawAppData() : Promise<void> {

        const appData = new Div('app-data', this.container);

        new Div('logo', appData);
    }

    /*
    **
    **
    */
    private async drawForm() : Promise<void> {

        this.form = new Form(this.content);

        //=====
        //EMAIL
        //=====

        this.email = this.form.add(new TextInput({
            label: 'Email'
        }));

        this.email.linkToErrorKey('email');
        this.email.input.detectAutoFill();

        //========
        //PASSWORD
        //========

        this.password = this.form.add(new PasswordInput({
            label: 'Password'
        }));

        this.password.linkToErrorKey('password');
        this.password.input.detectAutoFill();

        this.form.on('enter-down', this.onValid.bind(this));

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            const prerequisites = await Api.post('/login/prerequisites', {
                email: this.email.input.getValue()
            });

            let passwordHash: string;

            try {

                const split = prerequisites.split(':');
                const iterations = split[0];
                const saltHex = split[1];

                passwordHash = await PBKDF2.computePBKDF2HexHash(this.password.input.getValue(), saltHex, iterations, LoginPopup.HASH_DERIVATION_KEY_BYTE_LENGTH, LoginPopup.HASH_ALGORITHM);
            }
            catch(error) {

                passwordHash = Tools.sha256('~');
            };

            await Api.post('/login', {
                email: this.email.input.getValue(),
                passwordHash: passwordHash
            });

            await ClientLocation.get().api.checkAuth();
            
            this.hide();

        } catch(error: any) {

            if (error === 'access-denied@public-access-only') {
                await ClientLocation.get().api.checkAuth();
                this.hide();
                return;
            }

            this.form.displayError(error);
            this.validButton.unload();
        }
    }
}